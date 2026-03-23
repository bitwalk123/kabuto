import logging
from typing import Any

import numpy as np
import pandas as pd

from modules.agent_base import AgentBase
from structs.app_enum import ActionType, PositionType


class AgentCLI(AgentBase):
    """
    強化学習を利用せずに、アルゴリズムのみのエージェント
    （CLI 用）
    """
    BASE_COLUMNS = ["Timestamp", "Price", "Volume"]

    def __init__(self, code: str, dict_setting: dict[str, Any]) -> None:
        super().__init__(code, dict_setting)

    def addData(self, ts: float, price: float, volume: float) -> tuple[int, PositionType]:
        # 終了処理中はデータを処理しない
        if self._is_stopping or self.done:
            return ActionType.HOLD.value, PositionType.NONE

        # ティックデータから観測値を取得
        obs, dict_technicals = self.env.getObservation(ts, price, volume)

        # 現在の行動マスクを取得
        masks: np.ndarray = self.env.action_masks()

        # モデルによる行動予測
        action, _states = self.model.predict(obs, masks=masks)

        # メイン・スレッドへ通知する発注アクションを最優先
        position: PositionType = self.env.getCurrentPosition()
        # トレード後にまとめてデータフレームで出力するため
        for key, value in dict_technicals.items():
            self.dict_list_tech[key].append(value)

        # ---------------------------------------------------------------------
        # アクションによる環境の状態更新
        # 【注意】 リアルタイム用環境では step メソッドで観測値は返されない
        # ---------------------------------------------------------------------
        reward, terminated, truncated, info = self.env.step_realtime(action)
        if terminated or truncated:
            flag_name = "terminated" if terminated else "truncated"
            self.logger.info(f"{flag_name} フラグが立ちました。")
            self.done = True
        return action, position

    def cleanup(self) -> None:
        """
        スレッド終了前のクリーンアップ処理
        Trader.closeEvent から呼び出される想定（オプション）
        """
        self.logger.info(f"ワーカーのクリーンアップを開始します。")
        self._is_stopping = True

        # 必要に応じてリソースの解放処理を追加
        # 例：self.env.close() などがあれば呼び出す

        self.logger.info(f"ワーカーのクリーンアップが完了しました。")

    def forceRepay(self) -> None:
        """
        建玉返済の強制処理通知
        :return:
        """
        self.env.forceRepay()

    def resetEnv(self) -> None:
        # 環境のリセット
        self.obs, _ = self.env.reset()
        self.done = False
        self._is_stopping = False

        list_colname = self.BASE_COLUMNS.copy()
        self.list_obs_label = self.env.getObsList()
        self.model.updateObs(self.list_obs_label)
        list_colname.extend(self.list_obs_label)
        self.df_obs = pd.DataFrame({col: [] for col in list_colname})

    def setAutoPilot(self, flag: bool):
        self.model.setAutoPilot(flag)
