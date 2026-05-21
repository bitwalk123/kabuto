from typing import Any

import numpy as np

from models.abstract import AlgoTradeBase
from structs.app_enum import ActionType


class AlgoTrade(AlgoTradeBase):
    """疑似モデルのクラス"""
    """
    【仕様】クラス変数 MODEL_NAME にモデル名 = ファイル名（.py 除く）を保持する。
    """
    MODEL_NAME: str = "model_001"
    MODEL_REVISION: str = "0.0.1"

    def __init__(self):
        super().__init__()

    def predict(self, dict_obs, action_masks: np.ndarray) -> tuple[int, dict[str, Any]]:
        arr_signal = dict_obs["signal"]
        vwap_cross_golden = arr_signal[2]
        vwap_cross_dead = arr_signal[3]

        if 0.0 < vwap_cross_golden and self.can_execute(ActionType.BUY.value, action_masks):
            print("VWAP ゴールデンクロス")
            return ActionType.BUY.value, {"reason": "VWAP ゴールデンクロス"}

        if 0.0 < vwap_cross_dead and self.can_execute(ActionType.SELL.value, action_masks):
            print("VWAP デッドクロス")
            return ActionType.SELL.value, {"reason": "VWAP デッドクロス"}

        return ActionType.HOLD.value, {}

    def updateObs(self, list_obs_label):
        """
        観測値ラベルの更新（必須実装）

        疑似ロジックでは、観測値にラベルを付けておかないと、コーディングする側が間違える！
        【課題】
        ObservationManager クラスと整合・同期を取る仕組みを導入する必要がある。

        :param list_obs_label: 観測値ラベルのリスト
        :return:
        """
        pass
