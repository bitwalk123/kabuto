import gymnasium as gym
import numpy as np
import pandas as pd
import talib as ta
from enum import Enum


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
        self.df = df.reset_index(drop=True)  # Time, Price, Volume のみ
        self.last_volume = df.iloc[0]["Volume"]
        self.current_step = 0
        self.transman = TransactionManager()

        # obs: Price + ΔVolume + MA60 + STD60 + RSI60 + Z60 + one-hot(3)
        n_features = 1 + 1 + 4 + 3
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(n_features,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(len(ActionType))

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.transman.clearAll()
        self.last_volume = self.df.iloc[0]["Volume"]
        obs = self._get_observation()
        return obs, {}

    def step(self, n_action: int):
        # --- ウォームアップ期間は強制 HOLD ---
        if self.current_step < 60:
            action = ActionType.HOLD
        else:
            action = ActionType(n_action)

        reward = 0.0
        done = False

        price = self.df.at[self.current_step, "Price"]
        reward += self.transman.setAction(action, price)
        obs = self._get_observation()

        if self.current_step >= len(self.df) - 1:
            done = True

        self.current_step += 1

        dict_info = {"pnl_total": self.transman.pnl_total}
        return obs, reward, done, False, dict_info

    def _get_observation(self):
        row = self.df.iloc[self.current_step]
        price = row["Price"]

        # ΔVolume
        current_volume = row["Volume"]
        delta_volume = current_volume - self.last_volume
        log_volume = np.log1p(max(delta_volume, 0)).astype(np.float32)
        self.last_volume = current_volume

        # 過去60ティックのデータを切り出し
        start = max(0, self.current_step - 59)
        window = self.df.iloc[start:self.current_step + 1]

        if len(window) >= 60:
            close = window["Price"].values.astype(np.float64)
            ma60 = ta.SMA(close, timeperiod=60)[-1]
            rsi60 = ta.RSI(close, timeperiod=60)[-1]
            std60 = window["Price"].rolling(60).std().iloc[-1]
            z60 = (price - ma60) / std60 if std60 > 0 else 0.0
        else:
            ma60, std60, rsi60, z60 = 0.0, 0.0, 0.0, 0.0

        # PositionType → one-hot
        pos_onehot = np.eye(3)[self.transman.position.value].astype(np.float32)

        obs = np.array(
            [price, log_volume, ma60, std60, rsi60, z60],
            dtype=np.float32
        )
        obs = np.concatenate([obs, pos_onehot])
        return obs
