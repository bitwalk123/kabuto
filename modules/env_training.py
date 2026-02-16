from typing import override

import numpy as np
import pandas as pd

from modules.env import TradingEnv


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
        t: float = self.df.at[self.provider.step_current, "Time"]
        price: float = self.df.at[self.provider.step_current, "Price"]
        volume: float = self.df.at[self.provider.step_current, "Volume"]
        return t, price, volume

    @override
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        過去のティックデータを使うことを前提とした step 処理
        """
        info = dict()

        # ---------------------------------------------------------------------
        # アクションに対する報酬
        # ---------------------------------------------------------------------
        reward = self.reward_man.evalReward(action)

        # ---------------------------------------------------------------------
        # ステップ終了判定
        # ---------------------------------------------------------------------
        terminated = False
        truncated = False
        if len(self.df) - 1 <= self.provider.step_current:
            # ティックデータのステップ上限チェック
            reward += self.reward_man.forceRepay()
            truncated = True  # ← ステップ数上限による終了
            info["done_reason"] = "terminated: last_tick"
        elif self.provider.N_TRADE_MAX <= self.provider.n_trade:
            # 取引回数上限チェック
            reward += self.reward_man.forceRepay()
            truncated = True  # 取引回数上限による終了を明示
            info["done_reason"] = "terminated: max_trades"

        # 収益情報
        info["pnl_total"] = self.provider.pnl_total

        # ---------------------------------------------------------------------
        # 次の観測値
        # ---------------------------------------------------------------------
        # データフレームから次のティックデータを取得
        # t, price, volume = self._get_tick()
        self.provider.update(*self._get_tick())
        # モデルへ渡す観測値を取得
        obs, _ = self.obs_man.getObs()
        # step（行位置）をインクリメント
        self.provider.step_current += 1

        return obs, reward, terminated, truncated, info
