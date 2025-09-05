"""
TradingEnv (gymnasium.Env)

- Implements a gymnasium-compatible environment for single-JPY-stock 1-second tick data
- Enforces trading rules described by the user: 100-share lot, 1-yen tick, 1-tick slippage,
  no commissions, no averaging down (ナンピン禁止), max 1 open position (long or short),
  forced close at end of day, warm-up of 60 ticks where actions are forced to HOLD.

Usage:
  - Provide the environment with a pandas.DataFrame `df` containing columns at least:
      ['Time', 'Price', 'Volume', 'CumVolume']
    and (recommended precomputed) ['MA60','STD60','RSI60','Zscore60'].
  - If MA/STD/RSI/Zscore are missing, the env will compute running versions internally.
  - Reset accepts a starting index (default 0) so multiple episodes can be sampled from
    a long history.

Observation vector (np.float32):
  [price, ma60, std60, rsi60, zscore60, log1p_dvol, position_flag]
  - position_flag = 0 (no position), +1 (long/BUY), -1 (short/SELL)

Action space: Discrete(4) mapped to ActionType enum.

Reward:
  - On REPAY: realized PnL (in JPY) is added to reward.
  - Each tick while holding: reward += holding_share * (unrealized_pnl) * intraday_reward_scale
    (a small fraction to encourage/penalize mark-to-market)
  - Illegal / forbidden actions (ナンピン etc.) incur a fixed penalty (penalty_illegal_action).

Notes:
  - Slippage: order price will be executed at price +/- slippage_tick when opening/closing
  - Warm-up: until `warmup_n` ticks have been observed, all actions are forced to HOLD.

"""
from __future__ import annotations

import math
from enum import Enum, auto
from typing import Optional, Tuple, Dict, Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class ActionType(Enum):
    BUY = 0
    SELL = 1
    REPAY = 2
    HOLD = 3


class TradingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
            self,
            df: pd.DataFrame,
            start_idx: int = 0,
            lot_size: int = 100,
            tick_size: float = 1.0,
            slippage_ticks: int = 1,
            warmup_n: int = 60,
            intraday_reward_scale: float = 0.01,
            penalty_illegal_action: float = -1000.0,
            force_close_at: Optional[str] = None,  # time string like '15:00:00' if present
            compute_indicators_if_missing: bool = True,
    ) -> None:
        """
        df: must contain at least ['Time','Price','Volume','CumVolume'].
            Time should be parseable or already datetime; Price numeric; Volume numeric; CumVolume increasing cumulative volume.
        start_idx: starting row index for episodes by default 0.
        warmup_n: number of ticks to wait before allowing non-HOLD actions.
        intraday_reward_scale: fraction of unrealized PnL given as per-tick reward while holding.
        penalty_illegal_action: fixed penalty applied when rule violations happen.
        force_close_at: optional time (HH:MM:SS) to force close positions within an episode.
        compute_indicators_if_missing: whether to compute MA60/STD60/RSI60/Zscore60 if absent.
        """
        super().__init__()

        self.df = df.reset_index(drop=True).copy()
        self.start_idx = int(start_idx)
        self.lot_size = int(lot_size)
        self.tick_size = float(tick_size)
        self.slippage_ticks = int(slippage_ticks)
        self.warmup_n = int(warmup_n)
        self.intraday_reward_scale = float(intraday_reward_scale)
        self.penalty_illegal_action = float(penalty_illegal_action)
        self.force_close_at = force_close_at

        # Basic validation
        required = ["Time", "Price", "Volume", "CumVolume"]
        for c in required:
            if c not in self.df.columns:
                raise ValueError(f"DataFrame must contain column: {c}")

        # Ensure numeric types
        self.df["Price"] = pd.to_numeric(self.df["Price"])
        self.df["Volume"] = pd.to_numeric(self.df["Volume"])
        self.df["CumVolume"] = pd.to_numeric(self.df["CumVolume"])

        # Compute delta volume log1p feature in env as required
        self.df["dVolume"] = self.df["CumVolume"].diff().fillna(0).clip(lower=0)
        self.df["log1p_dVolume"] = np.log1p(self.df["dVolume"].values)

        # Compute indicators if missing
        if compute_indicators_if_missing:
            if "MA60" not in self.df.columns or "STD60" not in self.df.columns:
                self._compute_basic_running_stats(n=60)
            if "RSI60" not in self.df.columns:
                self._compute_rsi(n=60)
            if "Zscore60" not in self.df.columns:
                self._compute_zscore(n=60)

        # Build observation space
        # obs: [price, ma60, std60, rsi60, zscore60, log1p_dvol, position_flag]
        high = np.array([
            1e7,  # price upper (very high)
            1e7,  # ma
            1e7,  # std
            100.0,  # rsi
            1e7,  # zscore
            1e6,  # log1p dvol
            1.0,  # position flag
        ], dtype=np.float32)
        low = -high
        low[-1] = -1.0  # position flag can be -1/0/1
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Action space
        self.action_space = spaces.Discrete(len(ActionType))

        # Internal state
        self.idx = int(self.start_idx)
        self.position = 0  # 0 none, +1 long, -1 short
        self.entry_price: Optional[float] = None
        self.entry_idx: Optional[int] = None
        self.last_action: Optional[ActionType] = None
        self.done = False
        self.current_step_reward = 0.0

        # For episode accounting
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0

    # --------------------------- indicator helpers ---------------------------
    def _compute_basic_running_stats(self, n: int = 60) -> None:
        # rolling mean/std on price
        self.df["MA60"] = self.df["Price"].rolling(window=n, min_periods=1).mean()
        self.df["STD60"] = self.df["Price"].rolling(window=n, min_periods=1).std().fillna(0)

    def _compute_rsi(self, n: int = 60) -> None:
        # Simple RSI implementation using change in price
        delta = self.df["Price"].diff()
        up = delta.clip(lower=0).rolling(window=n, min_periods=1).mean()
        down = -delta.clip(upper=0).rolling(window=n, min_periods=1).mean()
        rs = up / (down.replace(0, np.nan))
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50.0)  # neutral when unknown
        self.df["RSI60"] = rsi

    def _compute_zscore(self, n: int = 60) -> None:
        rolling_mean = self.df["Price"].rolling(window=n, min_periods=1).mean()
        rolling_std = self.df["Price"].rolling(window=n, min_periods=1).std().replace(0, np.nan)
        z = (self.df["Price"] - rolling_mean) / rolling_std
        self.df["Zscore60"] = z.fillna(0.0)

    # --------------------------- gym API ------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        # options can provide 'start_idx' to sample different episodes
        if options and "start_idx" in options:
            self.idx = int(options["start_idx"])
        else:
            self.idx = int(self.start_idx)

        self.position = 0
        self.entry_price = None
        self.entry_idx = None
        self.last_action = None
        self.done = False
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0

        obs = self._get_obs()
        info = {"warmup": self.idx < self.warmup_n}
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.done:
            raise RuntimeError("Step called on done environment; call reset().")

        action_enum = ActionType(action)
        info: Dict[str, Any] = {}
        reward = 0.0

        # Warmup period: force HOLD
        if self.idx < self.warmup_n:
            if action_enum != ActionType.HOLD:
                reward += self.penalty_illegal_action * 0.1
            # advance time
            self.idx += 1
            obs = self._get_obs()
            self.last_action = ActionType.HOLD
            done = False
            return obs, reward, done, False, {"warmup": True}

        # Check forced close by time if requested
        if self.force_close_at is not None:
            cur_time = pd.to_datetime(self.df.at[self.idx, "Time"]).time()
            fc_time = pd.to_datetime(self.force_close_at).time()
            if cur_time >= fc_time:
                # Force a repay if position exists
                if self.position != 0:
                    # execute repay
                    repay_reward = self._execute_repay(
                        execution_price=self._get_slipped_price(self.df.at[self.idx, "Price"], close=True))
                    reward += repay_reward
                self.done = True
                obs = self._get_obs()
                return obs, reward, True, False, {"forced_close": True}

        # Enforce action legality (ナンピン禁止 rules)
        illegal = self._check_illegal_transition(self.last_action, action_enum)
        if illegal:
            reward += self.penalty_illegal_action
            # Do not execute illegal trade; treat as HOLD (safe behavior)
            action_enum = ActionType.HOLD

        # Map action to execution
        if action_enum == ActionType.BUY:
            # Open long if no position
            if self.position == 0:
                exec_price = self._get_slipped_price(self.df.at[self.idx, "Price"], long=True)
                self.position = 1
                self.entry_price = exec_price
                self.entry_idx = self.idx
            else:
                # If position exists, illegal due to no averaging
                reward += self.penalty_illegal_action

        elif action_enum == ActionType.SELL:
            # Open short if no position
            if self.position == 0:
                exec_price = self._get_slipped_price(self.df.at[self.idx, "Price"], long=False)
                self.position = -1
                self.entry_price = exec_price
                self.entry_idx = self.idx
            else:
                reward += self.penalty_illegal_action

        elif action_enum == ActionType.REPAY:
            # Close existing position
            if self.position != 0:
                exec_price = self._get_slipped_price(self.df.at[self.idx, "Price"], close=True)
                repay_reward = self._execute_repay(execution_price=exec_price)
                reward += repay_reward
            else:
                # REPAY when no position -> illegal
                reward += self.penalty_illegal_action

        elif action_enum == ActionType.HOLD:
            pass

        # Per-tick unrealized reward while holding: small fraction
        if self.position != 0 and self.entry_price is not None:
            current_price = float(self.df.at[self.idx, "Price"])
            unreal = (current_price - self.entry_price) * self.position * self.lot_size
            # Provide only a fraction to reduce noise
            reward += unreal * self.intraday_reward_scale
            self.unrealized_pnl = unreal

        # Advance time
        self.last_action = action_enum
        self.idx += 1

        # Check end of data
        if self.idx >= len(self.df):
            # If there is an open position close it at last price
            if self.position != 0 and self.idx - 1 >= 0:
                exec_price = self._get_slipped_price(self.df.at[min(self.idx - 1, len(self.df) - 1), "Price"],
                                                     close=True)
                repay_reward = self._execute_repay(execution_price=exec_price)
                reward += repay_reward
            self.done = True

        obs = self._get_obs()
        info.update({"realized_pnl": self.realized_pnl, "unrealized_pnl": self.unrealized_pnl})
        return obs, float(reward), bool(self.done), False, info

    def render(self, mode: str = "human"):
        if mode == "human":
            print(
                f"idx={self.idx}, price={self._cur_price()}, pos={self.position}, entry={self.entry_price}, realized_pnl={self.realized_pnl}")

    def close(self):
        pass

    # --------------------------- utilities ----------------------------------
    def _get_obs(self) -> np.ndarray:
        # Compose observation vector from current idx
        idx = min(self.idx, len(self.df) - 1)
        row = self.df.iloc[idx]
        price = float(row["Price"])
        ma = float(row.get("MA60", 0.0))
        std = float(row.get("STD60", 0.0))
        rsi = float(row.get("RSI60", 50.0))
        z = float(row.get("Zscore60", 0.0))
        log1p_dv = float(row.get("log1p_dVolume", 0.0))
        pos_flag = float(self.position)
        obs = np.array([price, ma, std, rsi, z, log1p_dv, pos_flag], dtype=np.float32)
        return obs

    def _cur_price(self) -> float:
        return float(self.df.at[min(self.idx, len(self.df) - 1), "Price"])

    def _get_slipped_price(self, price: float, long: Optional[bool] = None, close: bool = False) -> float:
        """
        Determine execution price given slippage and whether opening long/short or closing.
        - Opening long (BUY): pay price + slippage_ticks * tick_size
        - Opening short (SELL): receive price - slippage_ticks * tick_size
        - Closing: if closing long -> receive price - slippage; if closing short -> pay price + slippage.
        """
        p = float(price)
        s = self.slippage_ticks * self.tick_size
        if close:
            if self.position == 1:
                # closing long -> sell, worse by -slippage
                return p - s
            elif self.position == -1:
                # closing short -> buy back, worse by +slippage
                return p + s
            else:
                return p
        else:
            if long is True:
                return p + s
            elif long is False:
                return p - s
            else:
                return p

    def _execute_repay(self, execution_price: float) -> float:
        """Close the current position and return realized pnl (float).
        Also resets position state.
        """
        assert self.position != 0 and self.entry_price is not None
        exit_price = float(execution_price)
        pnl_per_share = (exit_price - self.entry_price) * self.position
        realized = pnl_per_share * self.lot_size
        self.realized_pnl += realized
        # reset
        self.position = 0
        self.entry_price = None
        self.entry_idx = None
        self.unrealized_pnl = 0.0
        return float(realized)

    def _check_illegal_transition(self, last_action: Optional[ActionType], new_action: ActionType) -> bool:
        # Implement rules from user spec regarding allowed transitions (ナンピン禁止 etc.)
        # last_action may be None (start of episode) -> treat as HOLD
        last = last_action if last_action is not None else ActionType.HOLD

        # Helper: whether there is position
        pos = self.position

        # The rules are enumerated in the user's prompt. We'll implement them explicitly.
        # Cases when last == HOLD
        if last == ActionType.HOLD:
            if pos == 0:
                # HOLD -> REPAY is forbidden
                if new_action == ActionType.REPAY:
                    return True
            else:
                # has position: HOLD -> BUY or SELL (i.e., opening another) forbidden
                if new_action == ActionType.BUY or new_action == ActionType.SELL:
                    return True
        # When last == BUY
        if last == ActionType.BUY:
            if new_action in (ActionType.BUY, ActionType.SELL):
                return True
        # When last == SELL
        if last == ActionType.SELL:
            if new_action in (ActionType.BUY, ActionType.SELL):
                return True
        # When last == REPAY
        if last == ActionType.REPAY:
            if new_action == ActionType.REPAY:
                return True
        # Additional rule: REPAY when no position is illegal
        if new_action == ActionType.REPAY and pos == 0:
            return True

        return False


# --------------------------- Example usage --------------------------------
# NOTE: put example under `if __name__ == '__main__':` when running locally.

if __name__ == "__main__":
    # Minimal example to instantiate and run random steps
    import datetime as dt

    # build toy dataframe
    n = 500
    times = pd.date_range("2025-09-01 09:00:00", periods=n, freq="S")
    prices = 1000 + np.cumsum(np.random.randn(n))
    volumes = np.random.poisson(10, size=n)
    cumvol = np.cumsum(volumes)
    df = pd.DataFrame({"Time": times, "Price": prices, "Volume": volumes, "CumVolume": cumvol})

    env = TradingEnv(df)
    obs, info = env.reset()
    done = False
    tot = 0.0
    while not done:
        a = env.action_space.sample()
        obs, r, done, trunc, info = env.step(a)
        tot += r
    print("Episode ended, realized pnl:", env.realized_pnl, "total reward:", tot)
