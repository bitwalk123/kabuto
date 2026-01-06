from structs.app_enum import SignalSign, ActionType, PositionType


class AlgoTrade:
    """
    強化学習モデルの代わりに、自作のアルゴリズムで取引するクラス
    """

    def __init__(self):
        self.list_obs_label = None
        self.idx_cross_1 = None
        self.idx_cross_2 = None
        self.idx_strength = None
        self.idx_losscut_1 = None
        self.idx_position = None

    def getListObs(self) -> list:
        return self.list_obs_label

    def predict(self, obs, masks) -> tuple[int, dict]:
        # 0. クロスシグナル 1
        cross_1 = SignalSign(int(obs[self.idx_cross_1]))
        # 1. クロスシグナル 2
        cross_2 = SignalSign(int(obs[self.idx_cross_2]))
        # 2. クロスシグナル強度
        strength = int(obs[self.idx_strength])
        # 3. ロスカット 1
        losscut_1 = int(obs[self.idx_losscut_1])
        # 4. ポジション（建玉）
        position = PositionType(int(obs[self.idx_position]))
        # ---------------------------------------------------------------------
        # シグナルの処理
        # ---------------------------------------------------------------------
        # 1. cross_1（建玉の有無に関係なく適用）
        if cross_1 in (SignalSign.POSITIVE, SignalSign.NEGATIVE):
            # エントリ方向の決定
            action_entry = (
                ActionType.BUY.value if cross_1 == SignalSign.POSITIVE else ActionType.SELL.value
            )
            action_exit = (
                ActionType.SELL.value if position == PositionType.LONG else ActionType.BUY.value
            )

            if position == PositionType.NONE:
                # エントリ
                if strength and masks[action_entry] == 1:
                    action = action_entry
                else:
                    action = ActionType.HOLD.value
            elif position in (PositionType.LONG, PositionType.SHORT):
                # エグジット
                if masks[action_exit] == 1:
                    action = action_exit
                else:
                    action = ActionType.HOLD.value
            else:
                raise TypeError(f"Unknown PositionType: {position}")
        # ---------------------------------------------------------------------
        # 2. cross_2（建玉が無い時のみ適用）
        elif cross_2 in (SignalSign.POSITIVE, SignalSign.NEGATIVE):

            if position == PositionType.NONE:
                action_entry = (
                    ActionType.BUY.value if cross_2 == SignalSign.POSITIVE else ActionType.SELL.value
                )

                if strength and masks[action_entry] == 1:
                    action = action_entry
                else:
                    action = ActionType.HOLD.value
            else:
                action = ActionType.HOLD.value
        # ---------------------------------------------------------------------
        # 3. losscut_1（単純ロスカット）
        else:
            if losscut_1:
                if position == PositionType.LONG:
                    action = ActionType.SELL.value if masks[ActionType.SELL.value] == 1 else ActionType.HOLD.value
                elif position == PositionType.SHORT:
                    action = ActionType.BUY.value if masks[ActionType.BUY.value] == 1 else ActionType.HOLD.value
                else:
                    action = ActionType.HOLD.value
            else:
                action = ActionType.HOLD.value
        # ---------------------------------------------------------------------
        return action, {}

    def updateObs(self, list_obs_label):
        self.list_obs_label = list_obs_label
        self.idx_cross_1 = self.list_obs_label.index("クロスS1")
        self.idx_cross_2 = self.list_obs_label.index("クロスS2")
        self.idx_strength = self.list_obs_label.index("クロ強")
        self.idx_losscut_1 = self.list_obs_label.index("ロス1")
        self.idx_position = self.list_obs_label.index("建玉")
