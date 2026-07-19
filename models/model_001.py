from typing import Any

import numpy as np

from funcs.conv import onehot_to_position
from models.abstract import AlgoTradeBase
from structs.app_enum import ActionType, PositionType


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
        ma_cross_golden: bool = arr_signal[0]
        ma_cross_dead: bool = arr_signal[1]
        vwap_cross_golden: bool = arr_signal[2]
        vwap_cross_dead: bool = arr_signal[3]
        flag_take_profit: bool = arr_signal[5]
        flag_losscut_consecutive_negative: bool = arr_signal[6]
        flag_losscut_simple: bool = arr_signal[7]

        position: PositionType = onehot_to_position(dict_obs["position"])

        if position == PositionType.NONE:
            '''
            # === エントリ ===
            if self.isAutoPilot():
                # MA ゴールデンクロスでエントリ
                if ma_cross_golden and self.can_execute(ActionType.BUY.value, action_masks):
                    return ActionType.BUY.value, {"reason": "MA ゴールデンクロス（買建）"}

                # MA デッドクロスでエントリ
                if ma_cross_dead and self.can_execute(ActionType.SELL.value, action_masks):
                    return ActionType.SELL.value, {"reason": "MA デッドクロス（売建）"}
            '''
            pass
        else:
            # === エグジット ===
            # MA ゴールデンクロスでエグジット
            if ma_cross_golden and self.can_execute(ActionType.BUY.value, action_masks):
                return ActionType.BUY.value, {"reason": "MA ゴールデンクロス（返済）"}

            # MA デッドクロスでエグジット
            if ma_cross_dead and self.can_execute(ActionType.SELL.value, action_masks):
                return ActionType.SELL.value, {"reason": "MA デッドクロス（返済）"}

            '''
            if flag_take_profit:
                # ドローダウン利確
                if position == PositionType.SHORT and self.can_execute(ActionType.BUY.value, action_masks):
                    return ActionType.BUY.value, {"reason": "ドローダウン利確（ショート）"}
                elif position == PositionType.LONG and self.can_execute(ActionType.SELL.value, action_masks):
                    return ActionType.SELL.value, {"reason": "ドローダウン利確（ロング）"}

            if flag_losscut_consecutive_negative:
                if position == PositionType.SHORT and self.can_execute(ActionType.BUY.value, action_masks):
                    return ActionType.BUY.value, {"reason": "連続含み損ロスカット（ショート）"}
                elif position == PositionType.LONG and self.can_execute(ActionType.SELL.value, action_masks):
                    return ActionType.SELL.value, {"reason": "連続含み損ロスカット（ロング）"}

            if flag_losscut_simple:
                if position == PositionType.SHORT and self.can_execute(ActionType.BUY.value, action_masks):
                    return ActionType.BUY.value, {"reason": "単純ロスカット（ショート）"}
                elif position == PositionType.LONG and self.can_execute(ActionType.SELL.value, action_masks):
                    return ActionType.SELL.value, {"reason": "単純ロスカット（ロング）"}
            '''

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
