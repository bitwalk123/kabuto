from typing import override

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium.utils import seeding

from modules.observatory import ObservationManager
from modules.feature_provider import FeatureProvider
from modules.remunerator import RewardManager
from structs.app_enum import ActionType, PositionType


class TradingEnv(gym.Env):
    """
    取引用環境クラス
    """

    def __init__(self, code: str = "7011"):
        super().__init__()
        # ウォームアップ期間
        self.n_warmup: int = 60

        # 現在の行位置
        self.step_current: int = 0

        # 特徴量プロバイダ
        self.provider = provider = FeatureProvider()
        # 売買管理クラス
        self.reward_man = RewardManager(provider, code)
        # 観測値管理クラス
        self.obs_man = ObservationManager(provider)

        # 観測空間
        n_feature = self.obs_man.n_feature
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_feature,),
            dtype=np.float32
        )
        # 行動空間
        self.action_space = gym.spaces.Discrete(len(ActionType))

    def action_masks(self) -> np.ndarray:
        """
        行動マスク
        【マスク】
        - ウォームアップ期間
        - ナンピン取引の禁止
        :return:
        """
        if self.step_current < self.n_warmup:
            # ウォームアップ期間 → 強制 HOLD
            return np.array([1, 0, 0], dtype=np.int8)
        elif self.reward_man.position == PositionType.NONE:
            # 建玉なし → 取りうるアクション: HOLD, BUY, SELL
            return np.array([1, 1, 1], dtype=np.int8)
        elif self.reward_man.position == PositionType.LONG:
            # 建玉あり LONG → 取りうるアクション: HOLD, SELL
            return np.array([1, 0, 1], dtype=np.int8)
        elif self.reward_man.position == PositionType.SHORT:
            # 建玉あり SHORT → 取りうるアクション: HOLD, BUY
            return np.array([1, 1, 0], dtype=np.int8)
        else:
            raise TypeError(f"Unknown PositionType: {self.reward_man.position}")

    def getTransaction(self) -> pd.DataFrame:
        return pd.DataFrame(self.reward_man.dict_transaction)

    def getObservation(self, ts: float, price: float, volume: float):
        """
        ティックデータから観測値を算出（デバッグ用）
        :param ts:
        :param price:
        :param volume:
        :return:
        """
        self.provider.update(ts, price, volume)
        # 観測値
        obs = self.obs_man.getObs(
            self.reward_man.getPL4Obs(),  # 含み損益
            self.reward_man.getPLMax4Obs(),  # 含み損益最大値
            self.reward_man.position,  # ポジション
        )
        return obs

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        """
        リセット
        :param seed:
        :param options:
        :return:
        """
        self.np_random, seed = seeding.np_random(seed)  # ← 乱数生成器を初期化
        self.step_current = 0
        self.reward_man.clear()
        obs = self.obs_man.getObsReset()
        return obs, {}

    def setData(self, ts: float, price: float, volume: float):
        """
        リアルタイム推論時には、step メソッドを呼ぶ前に本メソッドでデータを渡す。
        :param ts:
        :param price:
        :param volume:
        :return:
        """
        self.provider.update(ts, price, volume)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        アクションによるステップ処理
        :param action:
        :return:
        """
        info = dict()

        # ---------------------------------------------------------------------
        # 報酬
        # ---------------------------------------------------------------------
        reward = self.reward_man.evalReward(action)

        # ---------------------------------------------------------------------
        # 観測値
        # ---------------------------------------------------------------------
        obs = self.obs_man.getObs(
            self.reward_man.getPL4Obs(),  # 含み損益
            self.reward_man.getPLMax4Obs(),  # 含み損益最大値
            self.reward_man.position,  # ポジション
        )

        terminated = False
        truncated = False
        info["pnl_total"] = self.reward_man.pnl_total

        # ---------------------------------------------------------------------
        # 取引回数上限チェック
        # ---------------------------------------------------------------------
        if self.provider.n_trade_max <= self.provider.n_trade:
            reward += self.reward_man.forceRepay()
            truncated = True  # 取引回数上限による終了を明示
            info["done_reason"] = "terminated:max_trades"

        self.step_current += 1
        return obs, reward, terminated, truncated, info


class TrainingEnv(TradingEnv):
    """
    環境クラス
    過去のティックデータを使った学習、推論用
    """

    def __init__(self, df: pd.DataFrame, code: str = "7011"):
        super().__init__(code)
        self.df = df.reset_index(drop=True)  # Time, Price, Volume のみ

    def _get_tick(self) -> tuple[float, float, float]:
        """
        データフレームから 1 ステップ分のティックデータを読み込む
        :return:
        """
        t: float = self.df.at[self.step_current, "Time"]
        price: float = self.df.at[self.step_current, "Price"]
        volume: float = self.df.at[self.step_current, "Volume"]
        return t, price, volume

    @override
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        過去のティックデータを使うことを前提とした step 処理
        """
        info = dict()

        # ---------------------------------------------------------------------
        # データフレームからティックデータを取得
        # t, price, volume = self._get_tick()
        self.provider.update(*self._get_tick())
        # ---------------------------------------------------------------------

        # ---------------------------------------------------------------------
        # 報酬
        # ---------------------------------------------------------------------
        reward = self.reward_man.evalReward(action)

        # ---------------------------------------------------------------------
        # 観測値
        # ---------------------------------------------------------------------
        obs = self.obs_man.getObs(
            self.reward_man.getPL4Obs(),  # 含み損益
            self.reward_man.getPLMax4Obs(),  # 含み損益最大値
            self.reward_man.position,  # ポジション
        )

        terminated = False
        truncated = False

        # ---------------------------------------------------------------------
        # ティックデータのステップ上限チェック
        # ---------------------------------------------------------------------
        if len(self.df) - 1 <= self.step_current:
            reward += self.reward_man.forceRepay()
            truncated = True  # ← ステップ数上限による終了
            info["done_reason"] = "terminated: last_tick"

        # ---------------------------------------------------------------------
        # 取引回数上限チェック
        # ---------------------------------------------------------------------
        if self.provider.n_trade_max <= self.provider.n_trade:
            reward += self.reward_man.forceRepay()
            truncated = True  # 取引回数上限による終了を明示
            info["done_reason"] = "terminated: max_trades"

        self.step_current += 1
        info["pnl_total"] = self.reward_man.pnl_total

        return obs, reward, terminated, truncated, info
