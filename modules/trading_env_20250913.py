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
    # 売買管理クラス
    def __init__(self):
        self.reward_contract_bonus = 1.0  # 約定ボーナス（買建、売建、返済）
        self.reward_pnl_scale = 0.1  # 含み損益のスケール（比率に対する係数）
        self.penalty_rule = -1.0  # 売買ルール違反
        self.penalty_hold_small = -0.001  # 少しばかりの保持違反
        # 売買ルール違反カウンター
        self.penalty_count = 0  # 売買ルール違反ペナルティを繰り返すとカウントを加算

        self.action_pre = ActionType.HOLD
        self.position = PositionType.NONE
        self.price_entry = 0.0
        self.pnl_total = 0.0

    def clearAll(self):
        """
        初期状態に設定
        :return:
        """
        self.resetPosition()
        self.action_pre = ActionType.HOLD
        self.pnl_total = 0.0

    def resetPosition(self):
        """
        ポジション（建玉）をリセット
        :return:
        """
        self.position = PositionType.NONE
        self.price_entry = 0.0

    def setAction(self, action: ActionType, price: float) -> float:
        reward = 0.0
        if action == ActionType.HOLD:
            # ■■■ HOLD: 何もしない
            # 建玉があれば含み損益から報酬を付与、無ければ僅かなペナルティ
            reward += self.calc_reward_pnl(price)
            # 売買ルールを遵守した処理だったのでペナルティカウントをリセット
            self.penalty_count = 0
        elif action == ActionType.BUY:
            # ■■■ BUY: 信用買い
            if self.position == PositionType.NONE:
                # === 建玉がない場合 ===
                # 買建 (LONG)
                self.position = PositionType.LONG
                self.price_entry = price
                # 約定ボーナス付与（買建）
                reward += self.reward_contract_bonus
                # 売買ルールを遵守した処理だったのでペナルティカウントをリセット
                self.penalty_count = 0
            else:
                # ○○○ 建玉がある場合 ○○○
                # 建玉があるので、含み損益から報酬を付与
                reward += self.calc_reward_pnl(price)
                # ただし、建玉があるのに更に買建 (BUY) しようとしたので売買ルール違反ペナルティも付与
                self.penalty_count += 1
                reward += self.penalty_rule * self.penalty_count

        elif action == ActionType.SELL:
            # ■■■ SELL: 信用空売り
            if self.position == PositionType.NONE:
                # === 建玉がない場合 ===
                # 売建 (SHORT)
                self.position = PositionType.SHORT
                self.price_entry = price
                # 約定ボーナス付与（売建）
                reward += self.reward_contract_bonus
                # 売買ルールを遵守した処理だったのでペナルティカウントをリセット
                self.penalty_count = 0
            else:
                # ○○○ 建玉がある場合 ○○○
                # 建玉があるので、含み損益から報酬を付与
                reward += self.calc_reward_pnl(price)
                # ただし、建玉があるのに更に売建しようとしたので売買ルール違反ペナルティも付与
                self.penalty_count += 1
                reward += self.penalty_rule * self.penalty_count

        elif action == ActionType.REPAY:
            # ■■■ REPAY: 建玉返済
            if self.position != PositionType.NONE:
                # ○○○ 建玉がある場合 ○○○
                if self.position == PositionType.LONG:
                    # 実現損益（買建）
                    profit = price - self.price_entry
                else:
                    # 実現損益（売建）
                    profit = self.price_entry - price
                # ポジション状態をリセット
                self.resetPosition()
                # 総収益を更新
                self.pnl_total += profit
                # 報酬に収益を追加
                reward += profit
                # 約定ボーナス付与（返済）
                reward += self.reward_contract_bonus
                # 売買ルールを遵守した処理だったのでペナルティカウントをリセット
                self.penalty_count = 0
            else:
                # === 建玉がない場合 ===
                # 建玉がないのに建玉を返済しようとしたので売買ルール違反ペナルティを付与
                self.penalty_count += 1
                reward += self.penalty_rule * self.penalty_count
        else:
            raise ValueError(f"{action} is not defined!")

        self.action_pre = action
        return reward

    def calc_reward_pnl(self, price: float) -> float:
        """
        含み損益に self.reward_pnl_scale を乗じた報酬を算出
        ポジションが無い場合は微小なペナルティを付与
        :param price:
        :return:
        """
        if self.position == PositionType.LONG:
            # 含み損益（買建）× self.reward_pnl_scale
            return (price - self.price_entry) * self.reward_pnl_scale
        elif self.position == PositionType.SHORT:
            # 含み損益（売建）× self.reward_pnl_scale
            return (self.price_entry - price) * self.reward_pnl_scale
        else:
            # PositionType.NONE に対して僅かなペナルティ
            return self.penalty_hold_small


class TradingEnv(gym.Env):
    # 環境クラス
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df.reset_index(drop=True)  # Time, Price, Volume のみ

        self.last_volume = df.iloc[0]["Volume"]
        self.current_step = 0

        # 売買管理クラス
        self.transman = TransactionManager()

        # ウォームアップ期間
        self.period = 60

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
        # --- ウォームアップ期間 (self.period) は強制 HOLD ---
        if self.current_step < self.period:
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

        # 過去 self.period ティックのデータを切り出し
        start = max(0, self.current_step - self.period + 1)
        window = self.df.iloc[start:self.current_step + 1]

        if len(window) >= self.period:
            close = window["Price"].values.astype(np.float64)
            ma = ta.SMA(close, timeperiod=self.period)[-1]
            rsi = ta.RSI(close, timeperiod=self.period)[-1]
            sigma = window["Price"].rolling(self.period).std().iloc[-1]
            zscore = (price - ma) / sigma if sigma > 0 else 0.0
        else:
            ma, sigma, rsi, zscore = 0.0, 0.0, 0.0, 0.0

        # PositionType → one-hot
        pos_onehot = np.eye(3)[self.transman.position.value].astype(np.float32)

        obs = np.array(
            [price, log_volume, ma, sigma, rsi, zscore],
            dtype=np.float32
        )
        obs = np.concatenate([obs, pos_onehot])
        return obs
