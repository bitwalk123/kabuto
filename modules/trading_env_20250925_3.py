from __future__ import annotations
import numpy as np
import pandas as pd
from enum import Enum, auto
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

# -------------------------
# Action enum
# -------------------------
class ActionType(Enum):
    BUY = auto()   # 建玉の買い（ロング建て）
    SELL = auto()  # 建玉の売り（ショート建て）
    REPAY = auto() # 建玉の返済（損益確定）
    HOLD = auto()  # 何もしない / 建玉保持

# -------------------------
# Helper indicator functions (fallbacks)
# -------------------------
def rolling_ma(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=1).mean()

def rolling_std(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=1).std(ddof=0)

def rsi_series(price: pd.Series, n: int) -> pd.Series:
    # simple RSI without ta-lib; returns NaN for insufficient length
    diff = price.diff()
    up = diff.clip(lower=0.0)
    down = -diff.clip(upper=0.0)
    ma_up = up.rolling(n, min_periods=1).mean()
    ma_down = down.rolling(n, min_periods=1).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50.0)  # 中立値で埋める
    return rsi

def zscore(series: pd.Series, n: int) -> pd.Series:
    ma = rolling_ma(series, n)
    sd = rolling_std(series, n).replace(0, np.nan)
    zs = (series - ma) / sd
    return zs.fillna(0.0)

# -------------------------
# TradingEnv
# -------------------------
class TradingEnv(gym.Env):
    """
    TradingEnv for 1-stock intraday tick-by-tick simulation (1 sec ticks).
    - Expects a pandas.DataFrame with columns: ['Time', 'Price', 'Volume']
      * Time: float seconds (DO NOT convert to datetime)
      * Price: float (yen)
      * Volume: cumulative volume (float)
    - Lot size: 100 shares
    - Tick size: 1 yen
    - Slippage: 1 tick (1 yen)
    - No commission
    - Warm-up: first `warmup_n` ticks -> force HOLD
    """
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        lot_size: int = 100,
        tick: float = 1.0,
        slippage_ticks: int = 1,
        warmup_n: int = 60,
        unrealized_reward_scale: float = 0.01,
        illegal_action_penalty: float =  -1000.0,
        reward_scale: float = 1.0,
        debug: bool = False,
    ):
        super().__init__()
        assert {"Time", "Price", "Volume"}.issubset(df.columns), "df must contain Time, Price, Volume"

        # Keep raw df (do not modify user's timestamps)
        self.df_original = df.reset_index(drop=True).copy()
        self.n_steps = len(self.df_original)

        # env params
        self.lot_size = int(lot_size)
        self.tick = float(tick)
        self.slippage = slippage_ticks * self.tick
        self.warmup_n = int(warmup_n)
        self.unrealized_reward_scale = float(unrealized_reward_scale)
        self.illegal_action_penalty = float(illegal_action_penalty)
        self.reward_scale = float(reward_scale)
        self.debug = debug

        # Precompute internal features: dVolume_log, MA60, STD60, RSI60, Zscore60
        self._precompute_features()

        # Observation: we return a vector of floats:
        # [Price, MA60, STD60, RSI60, Zscore60, log_dvol, position, last_action]
        obs_low = np.array([0.0, -np.inf, 0.0, 0.0, -np.inf, 0.0, -1.0, 0.0], dtype=np.float32)
        obs_high = np.array([np.inf, np.inf, np.inf, 100.0, np.inf, np.inf, 1.0, 3.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Action space: discrete 4 (BUY, SELL, REPAY, HOLD)
        self.action_space = spaces.Discrete(len(ActionType))

        # Internal state
        self.reset()

    def _precompute_features(self):
        df = self.df_original
        # ΔVolume
        dvol = df["Volume"].diff().fillna(0.0).clip(lower=0.0)  # protect negatives if any
        log_dvol = np.log1p(dvol.values)

        price = df["Price"]
        # fallback technicals (use n=60)
        n = 60
        ma60 = rolling_ma(price, n).values
        std60 = rolling_std(price, n).fillna(0.0).values
        rsi60 = rsi_series(price, n).values
        z60 = zscore(price, n).values

        # store
        self.df_features = pd.DataFrame({
            "Time": df["Time"].values,
            "Price": df["Price"].values,
            "Volume": df["Volume"].values,
            "dvol": dvol.values,
            "log_dvol": log_dvol,
            "MA60": ma60,
            "STD60": std60,
            "RSI60": rsi60,
            "Z60": z60
        })

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict]:
        # reset pointers and position
        self.current_step = 0
        self.position = 0  # -1 (short), 0 (flat), +1 (long); max abs 1
        self.last_action = ActionType.HOLD
        self.entry_price: Optional[float] = None  # price at which current position was opened (including slippage)
        self.realized_pnl = 0.0
        self.terminated = False
        self.info = {}
        # return initial observation
        obs = self._get_observation(self.current_step)
        return obs.astype(np.float32), {}

    def _get_observation(self, idx: int) -> np.ndarray:
        row = self.df_features.iloc[idx]
        # Build observation vector
        # position and last_action encoded as floats
        last_action_idx = float(self.last_action.value - 1)  # 0..3
        obs = np.array([
            float(row["Price"]),
            float(row["MA60"]),
            float(row["STD60"]),
            float(row["RSI60"]),
            float(row["Z60"]),
            float(row["log_dvol"]),
            float(self.position),      # -1,0,1
            last_action_idx
        ], dtype=np.float32)
        return obs

    def _is_warmup(self) -> bool:
        return self.current_step < self.warmup_n

    def _illegal_action(self, action: ActionType) -> bool:
        # Implements the detailed forbidden transitions from spec
        # Evaluate based on current position and last_action
        pos = self.position
        la = self.last_action

        if pos == 0:
            # no position
            if la == ActionType.HOLD:
                # HOLD->REPAY forbidden
                if action == ActionType.REPAY:
                    return True
            # other combos (HOLD->BUY/SELL OK)
            if la == ActionType.REPAY:
                # REPAY->REPAY is forbidden
                if action == ActionType.REPAY:
                    return True
            # BUY/SELL from flat are OK
            return False
        else:
            # pos != 0 (have position either +1 or -1)
            if la == ActionType.HOLD:
                # HOLD -> BUY/SELL forbidden when already have a position (no pyramiding)
                if action in (ActionType.BUY, ActionType.SELL):
                    return True
            if la == ActionType.BUY:
                if action in (ActionType.BUY, ActionType.SELL):
                    return True
            if la == ActionType.SELL:
                if action in (ActionType.BUY, ActionType.SELL):
                    return True
            if la == ActionType.REPAY:
                if action == ActionType.REPAY:
                    return True
            return False

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Gymnasium step API: returns obs, reward, terminated, truncated, info
        We'll not use truncated here (return False).
        """
        if self.terminated:
            raise RuntimeError("Step called after termination. Call reset().")

        action = ActionType(action_idx + 1)  # Enum auto() started at 1
        row = self.df_features.iloc[self.current_step]
        price = float(row["Price"])
        reward = 0.0
        info = {}

        # Enforce warmup: force HOLD
        if self._is_warmup():
            # Force HOLD irrespective of model action
            if self.debug:
                info["warmup_forced"] = True
            self.last_action = ActionType.HOLD
            obs = self._get_observation(self.current_step)
            self.current_step += 1
            # no reward during warmup
            terminated = self.current_step >= self.n_steps
            return obs.astype(np.float32), 0.0, terminated, False, info

        # Check illegal action based on transition rules
        if self._illegal_action(action):
            # Apply penalty, do not change position nor last_action
            reward += self.illegal_action_penalty
            info["illegal_action"] = True
            if self.debug:
                print(f"[DEBUG] illegal action at step {self.current_step}: last_action={self.last_action}, pos={self.position}, action={action}")
            # advance time but no trade executed
            self.last_action = self.last_action  # unchanged
            obs = self._get_observation(self.current_step)
            self.current_step += 1
            terminated = self.current_step >= self.n_steps
            return obs.astype(np.float32), float(reward), terminated, False, info

        # Valid action handling
        # Helper: execute trade with slippage; lot * (execution_price - entry_price) -> PnL
        if action == ActionType.BUY:
            # Open long only if flat (pos==0) or allowed transition (but pyramiding forbidden)
            if self.position != 0:
                # Shouldn't happen due to _illegal_action check, but guard anyway
                reward += self.illegal_action_penalty
                info["illegal_action_guard"] = True
            else:
                exec_price = price + self.slippage  # pay slippage
                self.position = 1
                self.entry_price = exec_price
                if self.debug:
                    info["opened_long_at"] = exec_price

        elif action == ActionType.SELL:
            if self.position != 0:
                reward += self.illegal_action_penalty
                info["illegal_action_guard"] = True
            else:
                exec_price = price - self.slippage  # receive slightly worse for short
                self.position = -1
                self.entry_price = exec_price
                if self.debug:
                    info["opened_short_at"] = exec_price

        elif action == ActionType.REPAY:
            # Close any existing position; compute realized pnl
            if self.position == 0:
                # REPAY when no position — treat as illegal (spec: HOLD->REPAY when flat is forbidden)
                reward += self.illegal_action_penalty
                info["illegal_action_repay_flat"] = True
            else:
                # Execute repay at price +/- slippage depending on closing direction
                if self.position == 1:
                    # long -> sell to close => slippage negative for close (we assume same 1-tick cost)
                    exec_price = price - self.slippage
                else:
                    # short -> buy to close => pay slippage
                    exec_price = price + self.slippage
                # realized pnl per share:
                pnl_per_share = (exec_price - self.entry_price) * (1 if self.position == 1 else -1)
                # total pnl = pnl_per_share * lot_size
                realized = pnl_per_share * self.lot_size
                reward += realized * self.reward_scale
                self.realized_pnl += realized
                if self.debug:
                    info["repay_exec_price"] = exec_price
                    info["realized"] = realized
                # reset pos
                self.position = 0
                self.entry_price = None

        elif action == ActionType.HOLD:
            # Nothing executed; but we will give per-tick unrealized reward/penalty if holding
            pass

        # Per-tick unrealized reward while holding
        if self.position != 0 and self.entry_price is not None:
            # compute current mark-to-market using current tick price with no extra slippage
            current_price = price
            if self.position == 1:
                unrealized_per_share = (current_price - self.entry_price)
            else:
                unrealized_per_share = (self.entry_price - current_price)
            unrealized_total = unrealized_per_share * self.lot_size
            # add only a portion per tick to reduce variance (scale hyperparam)
            reward += unrealized_total * self.unrealized_reward_scale * self.reward_scale
            if self.debug:
                info["unrealized_total"] = unrealized_total

        # Update last_action and step pointer
        self.last_action = action
        obs = None
        self.current_step += 1
        terminated = self.current_step >= self.n_steps
        if not terminated:
            obs = self._get_observation(self.current_step)
        else:
            # End of data: force close any open position at final price (apply slippage)
            if self.position != 0 and self.entry_price is not None:
                final_row = self.df_features.iloc[-1]
                final_price = float(final_row["Price"])
                if self.position == 1:
                    exec_price = final_price - self.slippage
                else:
                    exec_price = final_price + self.slippage
                pnl_per_share = (exec_price - self.entry_price) * (1 if self.position == 1 else -1)
                realized = pnl_per_share * self.lot_size
                reward += realized * self.reward_scale
                self.realized_pnl += realized
                info["forced_close_realized"] = realized
                self.position = 0
                self.entry_price = None
            obs = self._get_observation(self.n_steps - 1)

        # wrap up
        return obs.astype(np.float32), float(reward), terminated, False, info

    def render(self, mode='human'):
        print(f"Step {self.current_step}/{self.n_steps} | Pos {self.position} | Entry {self.entry_price} | Realized PnL {self.realized_pnl}")

    def close(self):
        pass

# -------------------------
# Example usage snippet
# -------------------------
if __name__ == "__main__":
    # minimal example for quick smoke test
    # build toy DataFrame with 200 ticks
    times = np.arange(200.0)  # seconds as float (do not convert)
    prices = 2000.0 + np.cumsum(np.random.randn(len(times)) * 2.0)
    volumes = np.cumsum(np.random.poisson(10, size=len(times)).astype(float))
    df = pd.DataFrame({"Time": times, "Price": prices, "Volume": volumes})
    env = TradingEnv(df, debug=True)

    obs, _ = env.reset()
    total_reward = 0.0
    for t in range(len(df)):
        # naive policy: hold until warmup, then alternate BUY->REPAY->HOLD...
        if t < env.warmup_n:
            a = ActionType.HOLD
        else:
            # extremely naive random-ish policy for smoke test
            a = np.random.choice(list(ActionType))
        obs, r, done, trunc, info = env.step(a.value - 1)
        total_reward += r
        if done:
            break
    print("Total realized pnl:", env.realized_pnl)
    print("Total reward collected:", total_reward)
