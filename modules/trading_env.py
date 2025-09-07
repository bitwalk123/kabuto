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
        self.reward_rule = 1.0  # ルール適合報酬
        self.reward_rule_dbl = 2.0  # ルール適合報酬（大きめ）
        self.penalty_rule = -10.0  # ルール違反ペナルティ
        self.penalty_rule_dbl = -20.0  # ルール違反ペナルティ（厳しめ）

        self.reward_pnl_ratio = 0.1  # 含み損益に対する報酬比
        self.penalty_bit = -0.01  #

        self.position = PositionType.NONE
        self.price_entry = 0.0
        self.action_pre = ActionType.HOLD
        self.pnl_total = 0

    def clearPosition(self):
        self.position = PositionType.NONE
        self.price_entry = 0.0

    def clearAll(self):
        self.clearPosition()
        self.action_pre = ActionType.HOLD
        self.pnl_total = 0

    def has_position(self) -> bool:
        if self.price_entry > 0:
            return True
        else:
            return False

    def setAction(self, action: ActionType, price: float) -> float:
        reward = 0
        # 売買ルール
        if self.has_position():  # 建玉あり
            if self.action_pre == ActionType.HOLD:
                if action == ActionType.HOLD:
                    reward += self.reward_none
                elif action == ActionType.BUY:
                    reward += self.penalty_rule
                elif action == ActionType.SELL:
                    reward += self.penalty_rule
                elif action == ActionType.REPAY:
                    reward += self.reward_rule_dbl
                else:
                    pass
            elif self.action_pre == ActionType.BUY:
                if action == ActionType.HOLD:
                    reward += self.reward_none
                elif action == ActionType.BUY:
                    reward += self.penalty_rule_dbl
                elif action == ActionType.SELL:
                    reward += self.penalty_rule
                elif action == ActionType.REPAY:
                    reward += self.reward_rule_dbl
                else:
                    pass
            elif self.action_pre == ActionType.SELL:
                if action == ActionType.HOLD:
                    reward += self.reward_none
                elif action == ActionType.BUY:
                    reward += self.penalty_rule
                elif action == ActionType.SELL:
                    reward += self.penalty_rule_dbl
                elif action == ActionType.REPAY:
                    reward += self.reward_rule_dbl
                else:
                    pass
            elif self.action_pre == ActionType.REPAY:
                if action == ActionType.HOLD:
                    reward += self.reward_none
                elif action == ActionType.BUY:
                    reward += self.penalty_rule
                elif action == ActionType.SELL:
                    reward += self.penalty_rule
                elif action == ActionType.REPAY:
                    reward += self.penalty_rule_dbl
                else:
                    pass
        else:  # 建玉なし
            if self.action_pre == ActionType.HOLD:
                if action == ActionType.HOLD:
                    reward += self.penalty_bit
                elif action == ActionType.BUY:
                    reward += self.reward_rule_dbl
                elif action == ActionType.SELL:
                    reward += self.reward_rule_dbl
                elif action == ActionType.REPAY:
                    reward += self.penalty_rule
                else:
                    pass
            elif self.action_pre == ActionType.BUY:
                if action == ActionType.HOLD:
                    reward += self.reward_none
                elif action == ActionType.BUY:
                    reward += self.reward_rule
                elif action == ActionType.SELL:
                    reward += self.reward_rule
                elif action == ActionType.REPAY:
                    reward += self.penalty_rule
                else:
                    pass
            elif self.action_pre == ActionType.SELL:
                if action == ActionType.HOLD:
                    reward += self.reward_none
                elif action == ActionType.BUY:
                    reward += self.reward_rule
                elif action == ActionType.SELL:
                    reward += self.reward_rule
                elif action == ActionType.REPAY:
                    reward += self.penalty_rule
                else:
                    pass
            elif self.action_pre == ActionType.REPAY:
                if action == ActionType.HOLD:
                    reward += self.reward_none
                elif action == ActionType.BUY:
                    reward += self.reward_rule
                elif action == ActionType.SELL:
                    reward += self.reward_rule
                elif action == ActionType.REPAY:
                    reward += self.penalty_rule_dbl
                else:
                    pass

        # 一つ前のアクションを更新
        self.action_pre = action

        # 建玉損益
        if self.position == PositionType.LONG:
            pnl = price - self.price_entry
            if action == ActionType.REPAY:  # 利確
                reward += pnl
                self.pnl_total += pnl
                self.clearPosition()
            else:  # 含み損益
                reward += pnl * self.reward_pnl_ratio
        elif self.position == PositionType.SHORT:
            pnl = self.price_entry - price
            if action == ActionType.REPAY:  # 利確
                reward += pnl
                self.pnl_total += pnl
                self.clearPosition()
            else:  # 含み損益
                reward += pnl * self.reward_pnl_ratio
        elif self.position == PositionType.NONE:
            if action == ActionType.BUY:  # 買建
                self.position = PositionType.LONG
                self.price_entry = price
            elif action == ActionType.SELL:  # 売建（空売り）
                self.position = PositionType.SHORT
                self.price_entry = price
            else:
                pass
        else:
            pass

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
        # print(self.current_step, price, n_action, action, reward, "position:", self.transman.has_position())
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
