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
        self.reward_pnl_scale = 0.01  # 含み損益のスケール（比率に対する係数）

        self.position = PositionType.NONE
        self.price_entry = 0.0
        self.pnl_total = 0.0

    def clearPosition(self):
        self.position = PositionType.NONE
        self.price_entry = 0.0

    def clearAll(self):
        self.clearPosition()
        self.pnl_total = 0.0

    def setAction(self, action: ActionType, price: float) -> float:
        reward = 0
        if action == ActionType.HOLD:
            if self.position == PositionType.LONG:
                reward += (price - self.price_entry) * self.reward_pnl_scale
            elif self.position == PositionType.SHORT:
                reward += (self.price_entry - price) * self.reward_pnl_scale
            else:
                pass
        elif action == ActionType.BUY:
            if self.position == PositionType.NONE:
                self.position = PositionType.LONG
                self.price_entry = price
            else:
                pass
        elif action == ActionType.SELL:
            if self.position == PositionType.NONE:
                self.position = PositionType.SHORT
                self.price_entry = price
            else:
                pass
        elif action == ActionType.REPAY:
            if self.position == PositionType.LONG:
                profit = price - self.price_entry
                self.price_entry = 0.0
                self.position = PositionType.NONE
                self.pnl_total += profit
                reward += profit
            elif self.position == PositionType.SHORT:
                profit = self.price_entry - price
                self.price_entry = 0.0
                self.position = PositionType.NONE
                self.pnl_total += profit
                reward += profit
            else:
                pass
        else:
            raise ValueError(f"{action} is not defined!")

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
