from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Tuple, Optional, Any, List

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class ActionType(Enum):
    BUY = auto()  # 新規買い（+1建て）
    SELL = auto()  # 新規売り（-1建て）
    REPAY = auto()  # 返済（建玉クローズ）
    HOLD = auto()  # 何もしない（保有またはノーポジ維持）


@dataclass
class TradeState:
    position: int = 0  # -1, 0, +1 （100株を1単位とする）
    entry_price: Optional[float] = None
    realized_pnl: float = 0.0
    last_action: ActionType = ActionType.HOLD


class TradingEnv(gym.Env):
    """
    単一銘柄・1秒ティック前提のデイトレ環境。
    - 売買単位は常に100株、最大1単位まで。
    - 呼び値=1円、スリッページ=常に1ティック（=1円）を約定価格に反映。
    - 取引手数料は考慮しない。
    - ナンピン禁止（建玉保有中は新規方向のアクション BUY/SELL を禁止）。
    - ウォームアップ中(最初の warmup_ticks)は必ず HOLD（それ以外を出すと違反ペナルティ）。
    - 大引け（エピソード終了時）に未決済があれば強制返済。
    観測:
      [Price, MA60, STD60, RSI60, ZScore60, log1p(ΔVolume), position, time_left_ratio]
    """
    metadata = {"render_modes": []}

    def __init__(
            self,
            df: pd.DataFrame,
            price_col: str = "Price",
            vol_col: str = "Volume",
            time_col: str = "Time",
            ma_col: str = "MA60",
            std_col: str = "STD60",
            rsi_col: str = "RSI60",
            z_col: str = "ZScore60",
            lot_size: int = 100,
            tick_size: float = 1.0,
            slippage_ticks: int = 1,
            warmup_ticks: int = 60,
            invalid_action_penalty: float = -100.0,
            warmup_violation_penalty: float = -5.0,
            repay_when_flat_penalty: float = -10.0,
            reward_unrealized_mark_to_market: bool = False,  # True にすると含み損益も逐次報酬に反映
            seed: Optional[int] = None,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.price_col = price_col
        self.vol_col = vol_col
        self.time_col = time_col
        self.ma_col = ma_col
        self.std_col = std_col
        self.rsi_col = rsi_col
        self.z_col = z_col

        self.lot_size = lot_size
        self.tick_size = float(tick_size)
        self.slippage = float(slippage_ticks) * self.tick_size

        self.warmup_ticks = int(warmup_ticks)
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.warmup_violation_penalty = float(warmup_violation_penalty)
        self.repay_when_flat_penalty = float(repay_when_flat_penalty)
        self.reward_unrealized_mark_to_market = bool(reward_unrealized_mark_to_market)

        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        else:
            self.np_random = np.random

        # env 内で log1p(ΔVolume) を算出
        # 1秒ティックなので ΔVolume はその秒の出来高（= vol_col）
        vol = self.df[self.vol_col].astype(float).fillna(0.0).to_numpy()
        self.log1p_delta_vol = np.log1p(vol).astype(np.float32)

        # 行数/終端
        self.n = len(self.df)
        if self.n < self.warmup_ticks + 2:
            raise ValueError("データが短すぎます。ウォームアップ後に少なくとも数ティック必要です。")

        # アクション空間（4値）
        self.action_space = spaces.Discrete(4)

        # 観測空間：8次元（上記 docstring の順番）
        # Price/MA/STD/RSI/Zscore/log1pΔVol/position/time_left_ratio
        obs_low = np.array([
            0.0,  # Price （0未満にならない想定）
            -np.inf,  # MA
            0.0,  # STD
            0.0,  # RSI
            -np.inf,  # ZScore
            0.0,  # log1p(ΔVol) >= 0
            -1.0,  # position
            0.0,  # time_left_ratio
        ], dtype=np.float32)
        obs_high = np.array([
            np.inf,  # Price
            np.inf,  # MA
            np.inf,  # STD
            100.0,  # RSI（0-100）
            np.inf,  # ZScore
            np.inf,  # log1p(ΔVol)
            1.0,  # position
            1.0,  # time_left_ratio
        ], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self._t: int = 0
        self.state = TradeState()

    # ===== ユーティリティ =====
    def _row(self, t: int) -> pd.Series:
        return self.df.iloc[t]

    def _price(self, t: int) -> float:
        return float(self.df.at[t, self.price_col])

    def _obs(self, t: int) -> np.ndarray:
        row = self._row(t)
        time_left_ratio = float((self.n - 1 - t) / max(1, self.n - 1))
        obs = np.array([
            float(row[self.price_col]),
            float(row[self.ma_col]),
            float(row[self.std_col]),
            float(row[self.rsi_col]),
            float(row[self.z_col]),
            float(self.log1p_delta_vol[t]),
            float(self.state.position),
            time_left_ratio,
        ], dtype=np.float32)
        return obs

    def _allowed_actions(self) -> List[ActionType]:
        pos = self.state.position
        if self._t < self.warmup_ticks:
            return [ActionType.HOLD]  # ウォームアップ中は HOLD のみ
        if pos == 0:
            # ノーポジ：HOLD/BUY/SELL はOK、REPAYはNG
            return [ActionType.HOLD, ActionType.BUY, ActionType.SELL]
        else:
            # 建玉あり（+1 or -1）：HOLD/REPAYのみOK
            return [ActionType.HOLD, ActionType.REPAY]

    def _execute_trade(self, action: ActionType, px: float) -> Tuple[float, Optional[str]]:
        """
        約定処理と確定損益の更新。
        返り値: (step_realized_pnl, violation_message)
        """
        step_realized = 0.0
        violation = None
        pos = self.state.position

        # ウォームアップ制約
        if self._t < self.warmup_ticks and action is not ActionType.HOLD:
            violation = "warmup_violation"
            action = ActionType.HOLD  # 強制的に HOLD 扱い

        allowed = self._allowed_actions()
        if action not in allowed:
            # 返済禁止/ナンピン禁止/REPAY連発などはここでブロック
            violation = "invalid_action"
            action = ActionType.HOLD

        # ここから実行
        if action == ActionType.BUY:
            # 新規 +1 建て
            assert pos == 0
            self.state.position = 1
            # 買いは不利側にスリッページを乗せる
            self.state.entry_price = px + self.slippage

        elif action == ActionType.SELL:
            # 新規 -1 建て
            assert pos == 0
            self.state.position = -1
            # 売りは不利側にスリッページを乗せる
            self.state.entry_price = px - self.slippage

        elif action == ActionType.REPAY:
            if pos == 0:
                violation = violation or "repay_when_flat"
            else:
                # 返済約定価格（不利側スリッページ）
                if pos == 1:
                    exit_px = px - self.slippage  # ロング返済は売り：受け取り価格は低め
                    pnl = (exit_px - self.state.entry_price) * self.lot_size
                else:  # pos == -1
                    exit_px = px + self.slippage  # ショート返済は買い：支払い価格は高め
                    pnl = (self.state.entry_price - exit_px) * self.lot_size
                step_realized += pnl
                self.state.realized_pnl += pnl
                self.state.position = 0
                self.state.entry_price = None

        elif action == ActionType.HOLD:
            pass

        self.state.last_action = action
        return step_realized, violation

    def _mark_to_market(self, px: float) -> float:
        """含み損益（情報提供用／任意で報酬に加算可能）"""
        if self.state.position == 0 or self.state.entry_price is None:
            return 0.0
        if self.state.position == 1:
            # 現在値で即売却したと仮定（不利側スリッページ）
            exit_px = px - self.slippage
            return (exit_px - self.state.entry_price) * self.lot_size
        else:
            # 現在値で即買戻し（不利側スリッページ）
            exit_px = px + self.slippage
            return (self.state.entry_price - exit_px) * self.lot_size

    # ===== Gym API =====
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self._t = 0
        self.state = TradeState()
        obs = self._obs(self._t)
        info = {
            "action_mask": self._mask(),
            "position": self.state.position,
            "realized_pnl": self.state.realized_pnl,
            "unrealized_pnl": self._mark_to_market(self._price(self._t)),
            "t_index": self._t,
        }
        return obs, info

    def step(self, action_idx: int):
        action = list(ActionType)[action_idx]
        px = self._price(self._t)

        # 実行
        step_realized, violation = self._execute_trade(action, px)

        # 報酬
        reward = step_realized
        if violation == "invalid_action":
            reward += self.invalid_action_penalty
        elif violation == "warmup_violation":
            reward += self.warmup_violation_penalty
        elif violation == "repay_when_flat":
            reward += self.repay_when_flat_penalty

        # 含み損益を報酬に入れたい場合
        if self.reward_unrealized_mark_to_market:
            reward += self._mark_to_market(px)

        # 次ティックへ
        terminated = False
        truncated = False

        # 最終ティックに達する前に観測は次インデックスを返す
        if self._t >= self.n - 1:
            # （通常ここには来ない設計だが保険）
            terminated = True
        else:
            self._t += 1

        # 最終ティックに到達したら強制クローズ
        if self._t == self.n - 1:
            # 次の step 呼び出しで terminated になるので、ここで強制返済を実施
            if self.state.position != 0:
                px_last = self._price(self._t)
                # 強制返済（違反扱いなし、スリッページは通常どおり）
                if self.state.position == 1:
                    exit_px = px_last - self.slippage
                    pnl = (exit_px - self.state.entry_price) * self.lot_size
                else:
                    exit_px = px_last + self.slippage
                    pnl = (self.state.entry_price - exit_px) * self.lot_size
                self.state.realized_pnl += pnl
                reward += pnl
                self.state.position = 0
                self.state.entry_price = None

        # エピソード終了判定：観測を返した後、最終ティックで終了
        terminated = (self._t == self.n - 1)

        obs = self._obs(self._t)
        info = {
            "violation": violation,
            "action_executed": self.state.last_action.name,
            "action_mask": self._mask(),
            "position": self.state.position,
            "entry_price": self.state.entry_price,
            "realized_pnl": self.state.realized_pnl,
            "unrealized_pnl": self._mark_to_market(self._price(self._t)),
            "t_index": self._t,
            "time": self.df.at[self._t, self.time_col] if self.time_col in self.df.columns else None,
        }
        return obs, float(reward), terminated, truncated, info

    def _mask(self) -> np.ndarray:
        """エージェント用のアクションマスク（allowed=1, disallowed=0）。"""
        allowed = set(a.name for a in self._allowed_actions())
        mask = np.array(
            [
                1 if "BUY" in allowed else 0,
                1 if "SELL" in allowed else 0,
                1 if "REPAY" in allowed else 0,
                1 if "HOLD" in allowed else 0,
            ],
            dtype=np.int8,
        )
        return mask

    # gymnasium 互換の render は省略（必要ならログ出力等を実装）
    def render(self):
        pass
