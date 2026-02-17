from typing import Any

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

    def __init__(self, code: str, dict_setting: dict[str, Any]):
        super().__init__()
        # 特徴量プロバイダ
        self.provider = provider = FeatureProvider(dict_setting)
        # 売買管理クラス
        self.reward_man = RewardManager(provider, code)
        # 観測値管理クラス
        self.obs_man = ObservationManager(provider)

        # ウォームアップ期間
        self.n_warmup: int = provider.getPeriodWarmup()

        # 現在の行位置
        # self.provider.step_current = 0
        self.provider.setStepCurrent(0)

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
        position: PositionType = self.provider.getCurrentPosition()
        if self.provider.getStepCurrent() < self.n_warmup:
            # ウォームアップ期間 → 強制 HOLD
            return np.array([1, 0, 0], dtype=np.int8)
        elif position == PositionType.NONE:
            # 建玉なし → 取りうるアクション: HOLD, BUY, SELL
            return np.array([1, 1, 1], dtype=np.int8)
        elif position == PositionType.LONG:
            # 建玉あり LONG → 取りうるアクション: HOLD, SELL
            return np.array([1, 0, 1], dtype=np.int8)
        elif position == PositionType.SHORT:
            # 建玉あり SHORT → 取りうるアクション: HOLD, BUY
            return np.array([1, 1, 0], dtype=np.int8)
        else:
            raise TypeError(f"Unknown PositionType: {position}")

    def forceRepay(self) -> None:
        """
        建玉の強制返済
        :return:
        """
        self.reward_man.forceRepay()

    def getCurrentPosition(self) -> PositionType:
        """
        現在のポジションを返す
        :return:
        """
        return self.provider.getCurrentPosition()

    def getParams(self) -> dict[str, Any]:
        """
        調整可能？なパラメータを辞書で返す
        :return:
        """
        return self.provider.getSetting()

    def getTimestamp(self) -> float:
        # return self.provider.ts
        return self.provider.getTimestamp()

    def getTransaction(self) -> pd.DataFrame:
        #return pd.DataFrame(self.provider.dict_transaction)
        return pd.DataFrame(self.provider.getTransaction())

    def getObservation(
            self,
            ts: float,
            price: float,
            volume: float
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        観測値を取得（リアルタイム用）
        ティックデータから観測値を算出（デバッグ用）
        :param ts:
        :param price:
        :param volume:
        :return:
        """
        self.provider.update(ts, price, volume)
        # 観測値
        obs, dict_technicals = self.obs_man.getObs()
        return obs, dict_technicals

    def getObsList(self) -> list[str]:
        return self.obs_man.getObsList()

    def reset(
            self,
            seed: int | None = None,
            options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        リセット
        :param seed:
        :param options:
        :return:
        """
        self.np_random, seed = seeding.np_random(seed)  # ← 乱数生成器を初期化
        self.provider.clear()
        obs = self.obs_man.getObsReset()

        return obs, {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Gymnasium標準のstepメソッド（学習用）

        Note:
            このメソッドは過去データを使った学習・バックテスト用です。
            リアルタイム推論では step_realtime() を使用してください。

        Returns:
            observation: 次の観測値
            reward: 報酬
            terminated: エピソード終了フラグ
            truncated: エピソード打ち切りフラグ
            info: 追加情報
        """
        raise NotImplementedError(
            "標準のstepメソッドは未実装です。"
            "リアルタイム推論には step_realtime() を使用してください。"
            "学習用には TrainingEnv クラスの実装を検討してください。"
        )

    def step_realtime(self, action: int) -> tuple[float, bool, bool, dict[str, Any]]:
        """
        アクションによるステップ処理（リアルタイム用）

        Note:
            このメソッドは観測値を返しません。
            観測値は事前に getObservation() で取得済みという前提です。

        Args:
            action: 実行するアクション

        Returns:
            reward: 報酬
            terminated: エピソード終了フラグ（目標達成など）
            truncated: エピソード打ち切りフラグ（取引上限など）
            info: 追加情報（pnl_total, done_reasonなど）
        """
        info: dict[str, Any] = {}  # 型を明示

        # アクションに対する報酬
        reward = self.reward_man.evalReward(action)

        # ステップ終了判定
        terminated = False
        truncated = False

        # 取引回数上限チェック
        if self.provider.N_TRADE_MAX <= self.provider.getNTrade():
            reward += self.reward_man.forceRepay()
            truncated = True
            info["done_reason"] = "terminated:max_trades"

        # 収益情報
        info["pnl_total"] = self.provider.getPnLTotal()

        #self.provider.step_current += 1
        self.provider.setStepCurrentInc(1)
        return reward, terminated, truncated, info

    def openPosition(self, action_type: ActionType):
        if action_type == ActionType.BUY:
            self.provider.position_open(PositionType.LONG)
        elif action_type == ActionType.SELL:
            self.provider.position_open(PositionType.SHORT)
        else:
            raise TypeError(f"Unknown ActionType: {action_type}")

    def closePosition(self):
        self.provider.position_close()
