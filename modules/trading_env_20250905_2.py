"""
TradingEnv for gymnasium

- Implements a tick-based trading environment for single Japanese stock (100-share lot)
- Assumes input data is a pandas DataFrame with columns: ['Time', 'Price', 'Volume']
  * Volume is cumulative (累計出来高)
- Expects preprocessed columns (optional but recommended):
  * 'MA60', 'STD60', 'RSI60', 'ZSCORE60'
  If these are missing the env will compute them on the fly using pandas rolling.

- Special behaviour / constraints implemented:
  * Actions: BUY, SELL, REPAY, HOLD (Enum based)
  * ナンピン（ポジション追加）禁止：違反アクションには大きなペナルティを与える
  * 建玉は最大 1 単位（100株）まで
  * スリッページ = 1 円（常に不利に働く）
  * 手数料は考慮しない
  * 最初の 60 ティックはウォームアップ期間で必ず HOLD に強制される
  * 累計出来高の増分を np.log1p(ΔVolume) として observation に含める
  * 大引け（データ終端）では保有ポジションを強制クローズして確定損益を報酬として計上

- 報酬設計（既定）:
  * REPAY（ポジション解消）時に確定損益をそのまま加算
  * 含み益／含み損の一部を毎ティックで報酬に反映（unrealized_reward_fraction）
  * 違反アクションは penalty_illegal_action をマイナス報酬として付与

Usage:
  env = TradingEnv(df, lot_size=100)
  obs, info = env.reset()
  obs, reward, terminated, truncated, info = env.step(action)

"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from gymnasium import Env, spaces


class ActionType(enum.IntEnum):
    BUY = 0
    SELL = 1
    REPAY = 2
    HOLD = 3


@dataclass
class Position:
    side: int = 0  # 0 = none, +1 = long, -1 = short
    entry_price: Optional[float] = None


class TradingEnv(Env):
    """Gymnasium environment for tick-level day trading (single symbol)

    Parameters
    ----------
    data : pd.DataFrame
        Columns required: ['Time', 'Price', 'Volume'] (Volume is cumulative)
        Optional (recommended): 'MA60', 'STD60', 'RSI60', 'ZSCORE60'
    lot_size : int
        Number of shares per trade unit (default 100)
    slippage_ticks : int
        Slippage in ticks (1 tick == 1 yen) applied *adversely* to execution
    warmup : int
        Warmup ticks during which only HOLD is allowed (default 60)
    unrealized_reward_fraction : float
        Fraction of unrealized PnL awarded per tick while holding (default 0.1)
    penalty_illegal_action : float
        Penalty value for illegal actions (default -100000)
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
            self,
            data: pd.DataFrame,
            lot_size: int = 100,
            slippage_ticks: int = 1,
            warmup: int = 60,
            unrealized_reward_fraction: float = 0.1,
            penalty_illegal_action: float = -100000.0,
    ):
        super().__init__()
        self._orig_df = data.reset_index(drop=True).copy()
        self.df = self._orig_df.copy()
        self.lot_size = int(lot_size)
        self.slippage = float(slippage_ticks)  # in yen
        self.warmup = int(warmup)
        self.unrealized_fraction = float(unrealized_reward_fraction)
        self.penalty_illegal_action = float(penalty_illegal_action)

        # Precompute or validate indicators
        self._ensure_indicators()
        self._compute_delta_volume()

        # Observation vector composition (float32):
        # [Price, log1p_dvol, MA60, STD60, RSI60, ZSCORE60, position_side, entry_price_norm]
        obs_low = np.array([
            0.0,  # price lower bound
            -np.inf,  # log1p dvol
            -np.inf,  # MA60
            0.0,  # STD60
            0.0,  # RSI 0-100
            -np.inf,  # Zscore
            -1.0,  # position side: -1,0,1
            0.0,  # entry price normalized
        ], dtype=np.float32)

        obs_high = np.array([
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            100.0,
            np.finfo(np.float32).max,
            1.0,
            np.finfo(np.float32).max,
        ], dtype=np.float32)

        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.action_space = spaces.Discrete(len(ActionType))

        # Internal state
        self._pos = Position()
        self._idx = 0
        self._rng = np.random.default_rng()
        self._last_action = ActionType.HOLD
        self._terminated = False
        self._truncated = False
        self._accumulated_reward = 0.0
        self._realized_pnl_total = 0.0

    # -----------------------
    # Data preparation utils
    # -----------------------
    def _ensure_indicators(self) -> None:
        df = self.df
        # Moving average and std
        if "MA60" not in df.columns or "STD60" not in df.columns:
            df["MA60"] = df["Price"].rolling(window=60, min_periods=1).mean()
            df["STD60"] = df["Price"].rolling(window=60, min_periods=1).std(ddof=0).fillna(0.0)
        if "RSI60" not in df.columns:
            df["RSI60"] = self._calc_rsi(df["Price"], 60)
        if "ZSCORE60" not in df.columns:
            # avoid division by zero
            df["ZSCORE60"] = (df["Price"] - df["MA60"]) / (df["STD60"].replace(0, np.nan))
            df["ZSCORE60"] = df["ZSCORE60"].fillna(0.0)
        self.df = df

    @staticmethod
    def _calc_rsi(series: pd.Series, n: int) -> pd.Series:
        delta = series.diff()
        up = delta.clip(lower=0.0)
        down = -delta.clip(upper=0.0)
        ma_up = up.rolling(n, min_periods=1).mean()
        ma_down = down.rolling(n, min_periods=1).mean()
        rs = ma_up / ma_down.replace(0, np.nan)
        rsi = 100 - 100 / (1 + rs)
        rsi = rsi.fillna(50.0)  # neutral when undefined
        return rsi

    def _compute_delta_volume(self) -> None:
        # Volume is cumulative; compute difference and apply log1p
        vol = self.df["Volume"].astype(float)
        dvol = vol.diff().fillna(0.0).clip(lower=0.0)
        self.df["LOG1P_DVOLUME"] = np.log1p(dvol.values)

    # -----------------------
    # Core gym functions
    # -----------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._pos = Position()
        self._idx = 0
        self._last_action = ActionType.HOLD
        self._terminated = False
        self._truncated = False
        self._accumulated_reward = 0.0
        self._realized_pnl_total = 0.0

        obs = self._get_observation(self._idx)
        info: Dict[str, Any] = {"idx": self._idx}
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self._terminated or self._truncated:
            raise RuntimeError("Episode is done; call reset() before step().")

        # current row
        row = self.df.iloc[self._idx]
        price = float(row["Price"])  # market price at this tick

        action_enum = ActionType(int(action))

        # Enforce warmup: before warmup ticks, only HOLD allowed
        if self._idx < self.warmup and action_enum != ActionType.HOLD:
            # illegal — treat as HOLD and penalize
            reward = self.penalty_illegal_action
            self._accumulated_reward += reward
            self._last_action = ActionType.HOLD
            obs = self._get_observation(self._idx)
            info = {"illegal": True, "reason": "warmup"}
            return obs, reward, False, False, info

        # Check if action is illegal based on current position + last action (ナンピン禁止 rules)
        illegal, reason = self._is_action_illegal(action_enum)
        if illegal:
            reward = self.penalty_illegal_action
            self._accumulated_reward += reward
            # no state change on illegal action (treat as HOLD)
            self._last_action = ActionType.HOLD
            obs = self._get_observation(self._idx)
            info = {"illegal": True, "reason": reason}
            # advance time anyway
            self._idx += 1
            # If reached end, close any open position forcefully
            if self._idx >= len(self.df):
                terminal_reward = self._force_close_if_needed(at_end=True)
                reward += terminal_reward
                self._terminated = True
            return obs, reward, self._terminated, self._truncated, info

        # Apply action
        reward = 0.0
        info: Dict[str, Any] = {}

        if action_enum == ActionType.BUY:
            # Open long if no position exists
            assert self._pos.side == 0
            exec_price = price + self.slippage  # adverse slippage for buy
            self._pos.side = 1
            self._pos.entry_price = exec_price
            info["exec_price"] = exec_price

        elif action_enum == ActionType.SELL:
            # Open short if no position exists
            assert self._pos.side == 0
            exec_price = price - self.slippage  # adverse slippage for sell (receive less)
            self._pos.side = -1
            self._pos.entry_price = exec_price
            info["exec_price"] = exec_price

        elif action_enum == ActionType.REPAY:
            # Close existing position
            if self._pos.side == 0:
                # shouldn't happen due to illegal-action checks, but safe-guard
                reward = self.penalty_illegal_action
                info["repay"] = "no_position"
            else:
                exec_price = price - self.slippage * self._pos.side
                # For long (side=1), slippage added when buying, subtracted when selling to close
                # For short (side=-1), similar logic holds
                pnl_per_share = (exec_price - self._pos.entry_price) * self._pos.side
                realized = pnl_per_share * float(self.lot_size)
                reward += realized
                self._realized_pnl_total += realized
                info["exec_price"] = exec_price
                info["realized_pnl"] = realized
                # clear position
                self._pos = Position()
        elif action_enum == ActionType.HOLD:
            pass

        # Every tick: if holding a position, provide fraction of unrealized PnL as incremental reward
        if self._pos.side != 0:
            # current market price used for mark-to-market
            current_price = price
            unreal_pnl_per_share = (current_price - self._pos.entry_price) * self._pos.side
            unreal = unreal_pnl_per_share * float(self.lot_size)
            # fraction applied per tick
            tick_unreal_reward = unreal * self.unrealized_fraction
            reward += tick_unreal_reward
            info["unreal_pnl"] = unreal
            info["tick_unreal_reward"] = tick_unreal_reward

        # accumulate
        self._accumulated_reward += reward

        # advance index (time)
        self._last_action = action_enum
        self._idx += 1

        # if reached end, force close
        if self._idx >= len(self.df):
            terminal_reward = self._force_close_if_needed(at_end=True)
            reward += terminal_reward
            self._terminated = True

        obs = self._get_observation(min(self._idx, len(self.df) - 1))
        return obs, float(reward), self._terminated, self._truncated, info

    def _force_close_if_needed(self, at_end: bool = False) -> float:
        # If position exists, close at last known market price with slippage and record realized PnL
        if self._pos.side == 0:
            return 0.0
        idx = min(self._idx, len(self.df) - 1)
        price = float(self.df.iloc[idx]["Price"])
        exec_price = price - self.slippage * self._pos.side
        pnl_per_share = (exec_price - self._pos.entry_price) * self._pos.side
        realized = pnl_per_share * float(self.lot_size)
        self._realized_pnl_total += realized
        self._pos = Position()
        self._accumulated_reward += realized
        return realized

    def _is_action_illegal(self, action: ActionType) -> Tuple[bool, str]:
        # Apply the detailed transition rules described in the prompt
        # Use self._pos.side and self._last_action
        pos = self._pos.side
        last = self._last_action
        a = action

        # Helper flags
        if pos == 0:
            # No position
            if last == ActionType.HOLD:
                if a == ActionType.REPAY:
                    return True, "HOLD->REPAY_with_no_position"
            # REPAY->REPAY handled below
        else:
            # pos exists
            if last == ActionType.HOLD:
                # HOLD -> BUY/SELL forbidden when pos exists
                if a in (ActionType.BUY, ActionType.SELL):
                    return True, "HOLD->open_when_position_exists (nampin)"
        # If last was BUY
        if last == ActionType.BUY:
            if a in (ActionType.BUY, ActionType.SELL):
                return True, "BUY->BUY_or_SELL forbidden"
        if last == ActionType.SELL:
            if a in (ActionType.SELL, ActionType.BUY):
                return True, "SELL->SELL_or_BUY forbidden"
        if last == ActionType.REPAY:
            if a == ActionType.REPAY:
                return True, "REPAY->REPAY forbidden"

        # Disallow attempting to open a new position when one already exists
        if pos != 0 and a in (ActionType.BUY, ActionType.SELL):
            return True, "attempt_open_when_already_pos (nampin)"

        # Disallow REPAY when there's no position
        if pos == 0 and a == ActionType.REPAY:
            return True, "repay_without_position"

        return False, "ok"

    def _get_observation(self, idx: int) -> np.ndarray:
        row = self.df.iloc[idx]
        price = float(row["Price"])
        log1p_dvol = float(row["LOG1P_DVOLUME"])
        ma60 = float(row.get("MA60", np.nan))
        std60 = float(row.get("STD60", np.nan))
        rsi60 = float(row.get("RSI60", 50.0))
        zscore60 = float(row.get("ZSCORE60", 0.0))

        pos_side = float(self._pos.side)
        entry_price_norm = 0.0
        if self._pos.entry_price is not None:
            entry_price_norm = float(self._pos.entry_price)

        obs = np.array([
            price,
            log1p_dvol,
            ma60,
            std60,
            rsi60,
            zscore60,
            pos_side,
            entry_price_norm,
        ], dtype=np.float32)
        return obs

    def render(self, mode: str = "human") -> None:
        print(f"idx={self._idx}, pos={self._pos}, realized_pnl={self._realized_pnl_total:.1f}")

    def close(self) -> None:
        return


if __name__ == "__main__":
    # Quick smoke test
    import datetime

    # create fake data
    n = 200
    times = [datetime.datetime(2025, 1, 1, 9, 0, 0) + pd.Timedelta(seconds=i) for i in range(n)]
    prices = 1000 + np.cumsum(np.random.randn(n))
    volumes = np.cumsum(np.random.poisson(5, size=n))
    df = pd.DataFrame({"Time": times, "Price": prices, "Volume": volumes})

    env = TradingEnv(df)
    obs, info = env.reset()
    done = False
    total = 0.0
    steps = 0
    while not done and steps < n:
        # naive policy: HOLD until warmup, then randomly open/close
        if env._idx < env.warmup:
            a = ActionType.HOLD
        else:
            # choose mostly HOLD
            if env._pos.side == 0 and env._rng.random() < 0.02:
                a = ActionType.BUY
            elif env._pos.side == 1 and env._rng.random() < 0.05:
                a = ActionType.REPAY
            else:
                a = ActionType.HOLD
        obs, r, terminated, truncated, info = env.step(int(a))
        total += r
        steps += 1
        done = terminated or truncated
    print("smoke test total reward:", total)
