import gymnasium as gym
import numpy as np
from enum import Enum

import pandas as pd


class ActionType(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2
    REPAY = 3


class PositionType(Enum):
    NONE = 0
    LONG = 1
    SHORT = 2


class TransactionManager:
    def __init__(self):
        self.reward_none = 0.0  # 報酬なし
        self.reward_allowance = 0.02  # お小遣い
        self.reward_pnl_patio = 0.01  # 含み損益に対する報酬比
        self.penalty_avg_down = -0.01  # ナンピン・アクションのペナルティ
        self.penalty_some = -0.005  #

        self.position = PositionType.NONE
        self.entry_price = 0.0
        self.pnl_total = 0

    def clearPosition(self):
        self.position = PositionType.NONE
        self.entry_price = 0.0

    def clearAll(self):
        self.clearPosition()
        self.pnl_total = 0

    def setAction(self, action: ActionType, price: float) -> float:
        if action == ActionType.HOLD:
            if self.position == PositionType.LONG:
                reward = (price - self.entry_price) * self.reward_pnl_patio
            elif self.position == PositionType.SHORT:
                reward = (self.entry_price - price) * self.reward_pnl_patio
            else:
                reward = self.penalty_some
        elif self.position == PositionType.LONG:
            pnl = price - self.entry_price
            if action == ActionType.REPAY:
                reward = pnl
                self.pnl_total += reward
                self.clearPosition()
            else:
                reward = self.penalty_avg_down + pnl
        elif self.position == PositionType.SHORT:
            pnl = self.entry_price - price
            if action == ActionType.REPAY:
                reward = pnl
                self.pnl_total += reward
                self.clearPosition()
            else:
                reward = self.penalty_avg_down + pnl
        elif self.position == PositionType.NONE:
            if action == ActionType.BUY:
                self.position = PositionType.LONG
                self.entry_price = price
                reward = self.reward_allowance
            elif action == ActionType.SELL:
                self.position = PositionType.SHORT
                self.entry_price = price
                # reward = self.reward_none
                reward = self.reward_allowance
            elif action == ActionType.REPAY:
                # ペナルティ要検討
                reward = self.penalty_some
            else:
                # ありえないケース（念の為）
                reward = self.reward_none
        else:
            reward = self.reward_none

        return reward


class TradingEnv(gym.Env):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df
        self.selected_features = ["Price"]  # Volume は加工した特徴量として扱うので含めない
        self.last_volume = df.iloc[0]["Volume"]  # Volume 初期値を設定
        self.current_step = 0
        self.transman = TransactionManager()

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.selected_features) + 1 + 3,),  # Price + log_volume + one-hot(3)
            dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(len(ActionType))

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.transman.clearAll()
        obs = self._get_observation()
        return obs, {}

    def step(self, n_action: int):
        action = ActionType(n_action)
        reward = 0.0
        done = False

        price = self.df.at[self.current_step, "Price"]
        reward += self.transman.setAction(action, price)
        #print(self.current_step, price, n_action, action, reward)
        obs = self._get_observation()
        if self.current_step >= len(self.df) - 1:
            done = True

        self.current_step += 1

        dict_info = {
            "pnl_total": self.transman.pnl_total,
        }

        """
        obs: 行動後の状態（＝現在の観測）
        reward: そのステップで得られた報酬
        terminated: エピソードが「自然終了」したか（例：目標達成、時間切れ）
        truncated: エピソードが「強制終了」されたか（例：最大ステップ数到達）
        info: デバッグやログ用の追加情報（辞書型）
        """
        return obs, reward, done, False, dict_info

    def _get_observation(self):
        row = self.df.iloc[self.current_step]

        features = row[self.selected_features].values.astype(np.float32)

        # ΔVolume の計算
        current_volume = row["Volume"]
        delta_volume = current_volume - self.last_volume
        log_volume = np.log1p(max(delta_volume, 0)).astype(np.float32)
        self.last_volume = current_volume  # 更新

        # PositionType を数値に変換して追加（one-hot で表現）
        pos_onehot = np.eye(3)[self.transman.position.value].astype(np.float32)

        # 観測ベクトル = Price + ΔVolume + 現在のポジション
        obs = np.concatenate([features, [log_volume], pos_onehot])

        return obs
