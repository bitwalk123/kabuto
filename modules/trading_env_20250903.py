"""
TradingEnv: a simple Gymnasium environment for 1-second tick intraday trading (Japanese stock, 1 stock only).

Features / assumptions (per your requirements):
- Single Japanese stock, 1-second tick data (columns: Time, Price).
- Trade unit fixed at 100 shares.
- Tick size = 1 JPY.
- Slippage always 1 tick (adverse to trader).
- No commission.

Action space (Discrete(3)):
- 0: HOLD
- 1: BUY  (if currently 0 -> open long 100; if currently -100 -> buy to close short -> go to 0)
- 2: SELL (if currently 0 -> open short 100; if currently 100 -> sell to close long -> go to 0)

Position model: position is one of -100, 0, +100 (shares). Entry price recorded when opening a position.

Observation:
- Dict with 'price' (float) and 'position' (float: -100/0/100) represented as numpy arrays.

Reward:
- Reward is realized PnL in JPY for any closed trade in that step (qty * (exit - entry)).
- Unclosed PnL is NOT included in step reward (you can compute it from info if desired).

Gymnasium API compatibility: step() -> obs, reward, terminated, truncated, info
reset() -> obs, info

"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from gymnasium import Env, spaces


class TradingEnv(Env):
    """A small template trading environment for intraday 1-second tick data.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with at least columns ['Time', 'Price']. Index is ignored.
    start_index : int, optional
        Row index to start the episode from. If None, starts at 0.
    max_steps : int | None
        Optional maximum number of steps for the episode. If None uses until end of data.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        start_index: Optional[int] = None,
        max_steps: Optional[int] = None,
    ) -> None:
        assert isinstance(df, pd.DataFrame), "df must be a pandas DataFrame"
        required = {"Time", "Price"}
        if not required.issubset(set(df.columns)):
            raise ValueError(f"df must contain columns: {required}")

        # Work on a copy
        self.df = df.reset_index(drop=True).copy()
        self.n = len(self.df)

        # Action space: 0=HOLD,1=BUY,2=SELL
        self.action_space = spaces.Discrete(3)

        # Observation: price and current position
        # price range: [0, 1e7], position range [-100, 100]
        self.observation_space = spaces.Dict(
            {
                "price": spaces.Box(low=0.0, high=1e7, shape=(1,), dtype=np.float32),
                "position": spaces.Box(low=-100.0, high=100.0, shape=(1,), dtype=np.float32),
            }
        )

        # trading parameters
        self.unit = 100  # shares per trade
        self.tick = 1.0  # JPY tick
        self.slippage = 1.0  # always 1 tick adverse

        # episode control
        self.start_index = 0 if start_index is None else int(start_index)
        if not (0 <= self.start_index < self.n):
            raise ValueError("start_index out of bounds")
        self.max_steps = None if max_steps is None else int(max_steps)

        # internal state
        self.idx = None
        self.position = 0  # in shares: -100, 0, 100
        self.entry_price: Optional[float] = None  # price at which current position was opened
        self.cumulative_reward = 0.0
        self.steps = 0

    # ---- utility helpers ----
    def _current_price(self) -> float:
        return float(self.df.loc[self.idx, "Price"])

    def _execute_trade(self, action: int) -> Tuple[float, float]:
        """Execute trade implied by action at current price including slippage.

        Returns (realized_pnl, executed_price)
        """
        price = self._current_price()
        realized = 0.0
        executed_price = price

        # Determine outcome depending on current position
        if action == 1:  # BUY
            # If currently short (-unit), BUY -> close short
            if self.position == -self.unit:
                # buy to cover at price + slippage (worse for buyer)
                executed_price = price + self.slippage
                # PnL for short: (entry_price - exit_price) * qty
                assert self.entry_price is not None
                realized = (self.entry_price - executed_price) * self.unit
                self.position = 0
                self.entry_price = None
            elif self.position == 0:
                # open long
                executed_price = price + self.slippage
                self.position = self.unit
                self.entry_price = executed_price
            else:
                # already long -> HOLD semantics (no additional pyramiding)
                realized = 0.0

        elif action == 2:  # SELL
            # If currently long, SELL -> close long
            if self.position == self.unit:
                executed_price = price - self.slippage
                assert self.entry_price is not None
                realized = (executed_price - self.entry_price) * self.unit
                self.position = 0
                self.entry_price = None
            elif self.position == 0:
                # open short
                executed_price = price - self.slippage
                self.position = -self.unit
                self.entry_price = executed_price
            else:
                # already short -> HOLD semantics
                realized = 0.0

        else:  # HOLD
            realized = 0.0

        return realized, executed_price

    # ---- Gymnasium API ----
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment and return initial observation and info dict.

        You can pass 'start_index' in options to override the default start index.
        """
        if seed is not None:
            np.random.seed(seed)

        if options and "start_index" in options:
            s = int(options["start_index"])
            if not (0 <= s < self.n):
                raise ValueError("options['start_index'] out of bounds")
            self.start_index = s

        self.idx = self.start_index
        self.position = 0
        self.entry_price = None
        self.cumulative_reward = 0.0
        self.steps = 0

        obs = self._get_obs()
        info = {"start_index": self.start_index}
        return obs, info

    def step(self, action: int):
        """Step environment by one tick.

        Returns: obs, reward, terminated, truncated, info
        """
        assert self.idx is not None, "Call reset() before step()"
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # Execute trade (if any) at this tick's price
        realized_pnl, exec_price = self._execute_trade(action)

        # accumulate
        reward = float(realized_pnl)
        self.cumulative_reward += reward

        # advance pointer
        self.idx += 1
        self.steps += 1

        # termination conditions
        at_end = self.idx >= self.n
        truncated = False
        if self.max_steps is not None and self.steps >= self.max_steps:
            truncated = True
        terminated = at_end or truncated

        obs = self._get_obs() if not at_end else self._get_terminal_obs()

        info: Dict = {
            "position": self.position,
            "entry_price": self.entry_price,
            "executed_price": exec_price,
            "cumulative_reward": self.cumulative_reward,
            "time_idx": self.idx - 1,  # tick we just executed
        }

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        price = self._current_price()
        return {"price": np.array([price], dtype=np.float32), "position": np.array([float(self.position)], dtype=np.float32)}

    def _get_terminal_obs(self):
        # at terminal, we return last known price and position
        last_idx = min(self.n - 1, max(0, self.idx - 1))
        price = float(self.df.loc[last_idx, "Price"])
        return {"price": np.array([price], dtype=np.float32), "position": np.array([float(self.position)], dtype=np.float32)}

    def render(self, mode: str = "human"):
        if mode != "human":
            raise NotImplementedError("Only human render is supported")
        print(f"idx={self.idx}, time={self.df.loc[self.idx, 'Time'] if self.idx < self.n else 'END'}, price={self._current_price() if self.idx < self.n else 'N/A'}, position={self.position}, entry={self.entry_price}, cum_reward={self.cumulative_reward}")

    def close(self) -> None:
        return


# If run as a script, a tiny usage example (won't run during import)
if __name__ == "__main__":
    # create dummy tick data
    times = pd.date_range(start="2025-09-01 09:00:00", periods=10, freq="s")
    prices = np.array([1000, 1001, 1002, 1001, 1000, 999, 1000, 1001, 1002, 1001], dtype=float)
    df = pd.DataFrame({"Time": times, "Price": prices})

    env = TradingEnv(df)
    obs, info = env.reset()
    print("reset obs:", obs, info)

    done = False
    while not done:
        # random policy for demonstration
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print("step", action, reward, info)
        done = terminated

    print("episode finished, cumulative reward:", env.cumulative_reward)
