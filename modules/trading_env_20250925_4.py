# trading_env.py
from enum import Enum, auto
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


# -------------------------
# Action enum
# -------------------------
class ActionType(Enum):
    BUY = auto()  # 建玉の買い（ロング建て）
    SELL = auto()  # 建玉の売り（ショート建て）
    REPAY = auto()  # 建玉の返済（建玉があれば確定損益）
    HOLD = auto()  # 何もしない、または保持


# -------------------------
# TradingEnv
# -------------------------
class TradingEnv(gym.Env):
    """
    Gymnasium-compatible trading environment for single-JP-stock intraday ticks (1s ticks).

    Expected input DataFrame (index optional):
      - columns: ["Time", "Price", "Volume"]
      - Time: float seconds (keep as float; do NOT convert to datetime)
      - Price: float (yen)
      - Volume: float (cumulative total shares traded)
    """
    metadata = {"render.modes": ["human"]}

    def __init__(
            self,
            df: pd.DataFrame,
            warmup: int = 60,
            trade_size: int = 100,
            tick_size: float = 1.0,
            slippage_ticks: int = 1,
            illegal_penalty: float = -1000.0,
            unrealized_reward_rate: float = 0.01,  # 毎ティック含み益の一部を報酬化
    ):
        super().__init__()

        assert {"Time", "Price", "Volume"}.issubset(df.columns), "df must contain Time, Price, Volume"
        self.df_raw = df.reset_index(drop=True).copy()  # keep input intact but indexed 0..N-1
        self.N = len(self.df_raw)

        # config
        self.warmup = int(warmup)
        self.trade_size = int(trade_size)
        self.tick_size = float(tick_size)
        self.slippage = slippage_ticks * self.tick_size
        self.illegal_penalty = float(illegal_penalty)
        self.unrealized_reward_rate = float(unrealized_reward_rate)

        # Precompute rolling indicators (MA60, STD60, RSI60, ZSCORE60)
        self._compute_preprocessing_features()

        # Observation space: we'll return a vector with these elements:
        # [Price, MA60, STD60, RSI60, ZSCORE60, log1p_dVol, position (int: -1/0/1), time_frac]
        low = np.array([0.0, 0.0, 0.0, 0.0, -10.0, 0.0, -1.0, 0.0], dtype=np.float32)
        high = np.array([1e7, 1e7, 1e7, 100.0, 10.0, 1e6, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Action space: discrete mapped from ActionType
        self.action_map = [ActionType.BUY, ActionType.SELL, ActionType.REPAY, ActionType.HOLD]
        self.action_space = spaces.Discrete(len(self.action_map))

        # runtime state
        self.reset()

    def _compute_preprocessing_features(self):
        # compute rolling MA, STD, RSI (n=60), ZSCORE (n=60)
        n = 60
        s = self.df_raw["Price"]

        # MA60 and STD60
        self.df_raw["MA60"] = s.rolling(window=n, min_periods=1).mean()
        self.df_raw["STD60"] = s.rolling(window=n, min_periods=1).std(ddof=0).fillna(0.0)

        # RSI60 (simple version)
        delta = s.diff().fillna(0.0)
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.rolling(window=n, min_periods=1).mean()
        avg_loss = loss.rolling(window=n, min_periods=1).mean()
        rs = avg_gain / (avg_loss.replace(0, np.nan))
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50.0)  # neutral for early
        self.df_raw["RSI60"] = rsi

        # Z-score over window
        rolling_mean = s.rolling(window=n, min_periods=1).mean()
        rolling_std = s.rolling(window=n, min_periods=1).std(ddof=0).replace(0, np.nan)
        zscore = (s - rolling_mean) / rolling_std
        zscore = zscore.fillna(0.0)
        self.df_raw["ZSCORE60"] = zscore

        # ΔVolume and log1p ΔVolume (environment-level requirement)
        # Volume column is cumulative total; delta is current - previous
        #vol = self.df_raw["Volume"].fillna(method="ffill").fillna(0.0)
        vol = self.df_raw["Volume"].ffill().fillna(0.0)
        dvol = vol.diff().fillna(0.0).clip(lower=0.0)  # negative deltas shouldn't happen but guard
        self.df_raw["dVol"] = dvol
        self.df_raw["log1p_dVol"] = np.log1p(dvol)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, dict]:
        # pointer to current tick index
        self.idx = 0

        # position: 0 none, +1 long, -1 short
        self.position = 0
        self.entry_price = 0.0  # executed price of entry
        self.last_action: Optional[ActionType] = None

        # cumulative PnL tracking
        self.realized_pnl = 0.0

        # internal flag whether environment is done
        self.done = False

        # for logging trades
        self.trades = []

        # returns initial obs (warmup enforces HOLD)
        obs = self._make_observation(self.idx)
        info = {"idx": self.idx}
        return obs, info

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Returns: observation, reward, terminated, truncated, info
        We don't use truncated explicitly here; terminated == done.
        """
        if self.done:
            raise RuntimeError("Step called after done=True; call reset()")

        # map discrete -> ActionType
        action = self.action_map[int(action_idx)]
        info = {"action": action.name, "idx": self.idx}
        reward = 0.0
        illegal = False

        # Current tick
        row = self.df_raw.iloc[self.idx]
        price = float(row["Price"])
        time_sec = float(row["Time"])

        # Warmup: until idx >= warmup, force HOLD (no trades)
        if self.idx < self.warmup:
            # If agent attempted non-HOLD action during warmup, treat as illegal and apply penalty,
            # but force no state change (to keep behavior stable).
            if action is not ActionType.HOLD:
                reward += self.illegal_penalty
                illegal = True
            # per-tick small reward 0 for warmup
            next_idx = self.idx + 1
            self.idx = next_idx
            obs = self._make_observation(self.idx if self.idx < self.N else self.N - 1)
            terminated = self._maybe_force_close_or_end()
            return obs, reward, terminated, False, info

        # Check allowed transitions (per user's specification)
        # We'll implement a rule checker using current position and last_action.
        if not self._is_transition_allowed(self.last_action, action, self.position):
            reward += self.illegal_penalty
            illegal = True
            # Do not change position or execute trade when illegal (just penalize)
            # But still advance tick
            self.idx += 1
            obs = self._make_observation(self.idx if self.idx < self.N else self.N - 1)
            terminated = self._maybe_force_close_or_end()
            self.last_action = action
            return obs, reward, terminated, False, info

        # Execute allowed action
        # For BUY / SELL when no position: open position at price +/- slippage
        # For REPAY when position exists: close and realize PnL
        executed_price = price  # default (but will add slippage)
        if action is ActionType.BUY:
            if self.position == 0:
                executed_price = price + self.slippage  # buyer pays higher price
                self.position = 1
                self.entry_price = executed_price
                self.trades.append({"type": "BUY", "idx": self.idx, "price": executed_price})
            else:
                # Shouldn't reach here due to transition check (pyramiding forbidden)
                pass

        elif action is ActionType.SELL:
            if self.position == 0:
                executed_price = price - self.slippage  # seller receives lower price
                self.position = -1
                self.entry_price = executed_price
                self.trades.append({"type": "SELL", "idx": self.idx, "price": executed_price})
            else:
                # Shouldn't reach due to transition check
                pass

        elif action is ActionType.REPAY:
            if self.position != 0:
                # Closing: if long -> sell at price - slippage; if short -> buy at price + slippage
                if self.position == 1:
                    close_price = price - self.slippage
                    pnl_per_share = (close_price - self.entry_price)
                else:
                    close_price = price + self.slippage
                    pnl_per_share = (self.entry_price - close_price)
                realized = pnl_per_share * self.trade_size
                self.realized_pnl += realized
                reward += realized  # add realized PnL to reward
                self.trades.append({"type": "REPAY", "idx": self.idx, "price": close_price, "realized": realized})
                # clear position
                self.position = 0
                self.entry_price = 0.0
            else:
                # REPAY when no position is forbidden (transition checker handles this)
                pass

        elif action is ActionType.HOLD:
            # no trade; but we still give small per-tick unrealized reward/penalty below
            pass

        # per-tick unrealized reward: give a fraction of current unrealized PnL
        if self.position != 0:
            current_price = price
            if self.position == 1:
                mark_price = current_price - self.slippage  # assume slippage if closing now
                unrealized = (mark_price - self.entry_price) * self.trade_size
            else:
                mark_price = current_price + self.slippage
                unrealized = (self.entry_price - mark_price) * self.trade_size
            # give a fraction (can be negative)
            reward += self.unrealized_reward_rate * unrealized

        # advance time
        self.idx += 1
        obs = self._make_observation(self.idx if self.idx < self.N else self.N - 1)

        # update last action
        self.last_action = action

        # termination condition: end of data -> force close if position exists (apply penalty or close)
        terminated = self._maybe_force_close_or_end()

        return obs, float(reward), bool(terminated), False, info

    def _maybe_force_close_or_end(self) -> bool:
        # If we reached end of ticks -> if position exists, force REPAY (close) immediately
        if self.idx >= self.N:
            # force-close (大引けで強制クローズ)
            if self.position != 0:
                # use last price available as closing price (index N-1)
                last_row = self.df_raw.iloc[self.N - 1]
                price = float(last_row["Price"])
                if self.position == 1:
                    close_price = price - self.slippage
                    pnl_per_share = (close_price - self.entry_price)
                else:
                    close_price = price + self.slippage
                    pnl_per_share = (self.entry_price - close_price)
                realized = pnl_per_share * self.trade_size
                self.realized_pnl += realized
                self.trades.append(
                    {"type": "FORCED_REPAY", "idx": self.N - 1, "price": close_price, "realized": realized})
                # set position to zero
                self.position = 0
                self.entry_price = 0.0
                # We *could* add the realized PnL to final reward, but gym step already returned last reward.
                # For clarity, we don't retroactively modify previous step reward; user can read realized_pnl in info.
            self.done = True
            return True
        return False

    def _is_transition_allowed(self, last_action: Optional[ActionType], action: ActionType, position: int) -> bool:
        """
        Implements the transition table specified by the user. Returns True if allowed.
        """
        # If no last_action (start), allow:
        if last_action is None:
            # But REPAY at start (no pos) is forbidden
            if action is ActionType.REPAY and position == 0:
                return False
            return True

        # When no position (position == 0), rules:
        if position == 0:
            if last_action is None:
                last = None
            else:
                last = last_action
            # From HOLD -> REPAY is forbidden (when no position)
            if last == ActionType.HOLD and action == ActionType.REPAY:
                return False
            # From REPAY -> REPAY is forbidden (since no pos)
            if last == ActionType.REPAY and action == ActionType.REPAY:
                return False
            # REPAY -> BUY/SELL OK
            # HOLD -> BUY/SELL/ HOLD OK
            return True

        # When there is a position (position != 0)
        # From HOLD -> BUY/SELL forbidden (no pyramiding)
        if last_action == ActionType.HOLD and (action == ActionType.BUY or action == ActionType.SELL):
            return False

        # If last_action was BUY:
        if last_action == ActionType.BUY:
            if action in (ActionType.BUY, ActionType.SELL):
                return False
            return True  # REPAY / HOLD ok

        # If last_action was SELL:
        if last_action == ActionType.SELL:
            if action in (ActionType.BUY, ActionType.SELL):
                return False
            return True  # REPAY / HOLD ok

        # If last_action was REPAY:
        if last_action == ActionType.REPAY:
            if action == ActionType.REPAY:
                return False
            return True  # BUY/SELL/HOLD ok

        # default allow
        return True

    def _make_observation(self, idx: int) -> np.ndarray:
        # ensure idx is in-bounds
        if idx < 0:
            idx = 0
        if idx >= self.N:
            idx = self.N - 1
        row = self.df_raw.iloc[idx]

        price = float(row["Price"])
        ma60 = float(row.get("MA60", 0.0))
        std60 = float(row.get("STD60", 0.0))
        rsi60 = float(row.get("RSI60", 50.0))
        zscore60 = float(row.get("ZSCORE60", 0.0))
        log1p_dVol = float(row.get("log1p_dVol", 0.0))

        # position encoded as -1/0/1
        position = float(self.position)

        # time fraction within day (just normalized time to [0,1] for observation) but still store original Time elsewhere
        time_min = float(self.df_raw["Time"].min())
        time_max = float(self.df_raw["Time"].max())
        time_frac = 0.0
        if time_max > time_min:
            time_frac = (float(row["Time"]) - time_min) / (time_max - time_min)

        obs = np.array([
            price,
            ma60,
            std60,
            rsi60,
            zscore60,
            log1p_dVol,
            position,
            time_frac
        ], dtype=np.float32)
        return obs

    def render(self, mode="human"):
        print(f"idx={self.idx}, pos={self.position}, entry={self.entry_price}, realized_pnl={self.realized_pnl}")

    def close(self):
        pass

    # utility: get trade log and realized pnl
    def get_trade_log(self):
        return self.trades

    def get_realized_pnl(self):
        return self.realized_pnl


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    # small smoke-test using synthetic data
    np.random.seed(0)
    T = 500
    times = np.arange(T).astype(float)  # seconds as floats
    prices = 2000 + np.cumsum(np.random.normal(0, 1.0, size=T))  # random walk near 2000
    volumes = np.cumsum(np.random.poisson(5, size=T)).astype(float)

    df = pd.DataFrame({"Time": times, "Price": prices, "Volume": volumes})

    env = TradingEnv(df)
    obs, info = env.reset()
    total_reward = 0.0
    for step in range(1, len(df)):
        # naive random policy for smoke test (but respects action space)
        a = env.action_space.sample()
        obs, r, terminated, truncated, info = env.step(a)
        total_reward += r
        if terminated:
            break
    print("Total realized pnl:", env.get_realized_pnl())
    print("Total reward (sum):", total_reward)
    print("Trades:", env.get_trade_log())
