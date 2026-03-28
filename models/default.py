import os
from typing import Any

from models.abstract import AlgoTradeBase
from structs.app_enum import PositionType, ActionType


class AlgoTrade(AlgoTradeBase):
    """疑似モデルのクラス"""
    """
    【仕様】クラス変数 MODEL_NAME にモデル名 = ファイル名（.py 除く）を保持する。
    """
    MODEL_NAME: str = "default"
    MODEL_REVISION: str = "0.0.1"

    def __init__(self):
        super().__init__()
        # 観測値のインデックス
        self.idx_cross_1: int | None = None
        self.idx_cross_2: int | None = None
        self.idx_losscut_1: int | None = None
        self.idx_losscut_2: int | None = None
        self.idx_takeprofit_1: int | None = None
        self.idx_position: int | None = None

    def predict(self, obs, masks) -> tuple[int, dict[str, Any]]:
        # --- 観測値の取り出し ---
        # 1. クロスシグナル 1 [-1, 0, 1]
        cross_1 = int(obs[self.idx_cross_1])
        # 2. クロスシグナル 2 [0, 1], ゴールデン・クロスでエントリのみ
        cross_2 = int(obs[self.idx_cross_2])
        # 3. クロスシグナル 3 [-1, 0], デッド・クロスでエントリのみ
        cross_3 = int(obs[self.idx_cross_3])
        # 4. ロスカット 1 [0, 1]
        losscut_1 = int(obs[self.idx_losscut_1])
        # 5. ロスカット 2 [0, 1]
        losscut_2 = int(obs[self.idx_losscut_2])
        # 6. 利確 1 [0, 1]
        takeprofit_1 = int(obs[self.idx_takeprofit_1])
        # 7. ポジション（建玉） [SHORT, NONE, LONG]
        position = PositionType(int(obs[self.idx_position]))

        # --- エグジット判定 ---
        # 1. 継続（HOLD）条件を先に判定して早期リターン（ガード句）
        # ポジションとシグナルが一致している場合は、何もしない（HOLD）
        if (position == PositionType.LONG and cross_1 > 0) or \
                (position == PositionType.SHORT and cross_1 < 0):
            return ActionType.HOLD.value, {}

        # 2. エグジット判定が必要なシグナルがあるか確認
        # いずれかのフラグが立っている場合のみ処理を続行
        has_signal = any((cross_1, losscut_1, losscut_2, takeprofit_1))
        if has_signal:
            exit_act = self.exit_action(position)
            # 有効なアクションかつ実行可能ならそのアクションを返す
            if exit_act and self.can_execute(exit_act, masks):
                return exit_act, {}

        # 3. クロスシグナルによる自動エントリー ---
        if self.autopilot and position == PositionType.NONE:
            # a. ゴールデンクロス
            if cross_1 > 0:  # クロスS1: MA1 が VWAP を上抜け
                if self.can_execute(ActionType.BUY.value, masks):
                    # print("reason: golden_cross_1")
                    return ActionType.BUY.value, {'reason': 'golden_cross_1'}
            if cross_2 > 0:  # クロスS2: MA1 が VWAP上バンド を上抜け
                if self.can_execute(ActionType.BUY.value, masks):
                    # print("reason: golden_cross_2")
                    return ActionType.BUY.value, {'reason': 'golden_cross_2'}

            # b. デッドクロス
            if cross_1 < 0:  # クロスS1: MA1 が VWAP を下抜け
                if self.can_execute(ActionType.SELL.value, masks):
                    # print("reason: dead_cross_1")
                    return ActionType.SELL.value, {'reason': 'dead_cross_1'}
            if cross_3 < 0:  # クロスS3: MA1 が VWAP下バンド を下抜け
                if self.can_execute(ActionType.SELL.value, masks):
                    # print("reason: dead_cross_3")
                    return ActionType.SELL.value, {'reason': 'dead_cross_3'}

        # 4. デフォルトは HOLD
        return ActionType.HOLD.value, {}

    def updateObs(self, list_obs_label):
        """
        疑似ロジックでは、観測値にラベルを付けておかないと、コーディングする側が間違える！
        【課題】
        ObservationManager クラスと整合・同期を取る仕組みを導入する必要がある。
        :param list_obs_label:
        :return:
        """
        self.list_obs_label = list_obs_label
        self.idx_cross_1 = self.list_obs_label.index("クロスS1")
        self.idx_cross_2 = self.list_obs_label.index("クロスS2")
        self.idx_cross_3 = self.list_obs_label.index("クロスS3")
        self.idx_losscut_1 = self.list_obs_label.index("ロス1")
        self.idx_losscut_2 = self.list_obs_label.index("ロス2")
        self.idx_takeprofit_1 = self.list_obs_label.index("利確1")
        self.idx_position = self.list_obs_label.index("建玉")
