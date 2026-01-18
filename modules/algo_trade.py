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
        self.idx_fluc = None
        self.idx_losscut_1 = None
        self.idx_takeprofit_1 = None
        self.idx_position = None

    def getListObs(self) -> list:
        return self.list_obs_label

    def can_execute(self, action, masks):
        return masks[action] == 1

    def opposite_action(self, position: PositionType):
        if position == PositionType.LONG:
            return ActionType.SELL.value
        if position == PositionType.SHORT:
            return ActionType.BUY.value
        return None

    def entry_action(self, signal, strength, fluctuation, masks):
        action = (
            ActionType.BUY.value
            if signal == SignalSign.POSITIVE
            else ActionType.SELL.value
        )

        if not strength or not self.can_execute(action, masks):
            return ActionType.HOLD.value

        return ActionType.HOLD.value if fluctuation else action

    def predict(self, obs, masks) -> tuple[int, dict]:
        # --- 観測値の取り出し ---
        # 1. クロスシグナル 1
        cross_1 = SignalSign(int(obs[self.idx_cross_1]))
        # 2. クロスシグナル 2
        cross_2 = SignalSign(int(obs[self.idx_cross_2]))
        # 3. クロスシグナル強度
        strength = int(obs[self.idx_strength])
        # 4. 乱高下
        fluctuation = int(obs[self.idx_fluc])
        # 5. ロスカット 1
        losscut_1 = int(obs[self.idx_losscut_1])
        # 6. ロスカット 2
        losscut_2 = int(obs[self.idx_losscut_2])
        # 7. 利確 1
        takeprofit_1 = int(obs[self.idx_takeprofit_1])
        # 8. ポジション（建玉）
        position = PositionType(int(obs[self.idx_position]))

        # デフォルトは HOLD
        action = ActionType.HOLD.value

        # ------------------------------------------------------------
        # 1. cross_1（建玉の有無に関係なく適用）
        # ------------------------------------------------------------
        if cross_1 in (SignalSign.POSITIVE, SignalSign.NEGATIVE):

            if position == PositionType.NONE:
                action = self.entry_action(cross_1, strength, fluctuation, masks)

            else:
                exit_act = self.opposite_action(position)
                if exit_act is not None and self.can_execute(exit_act, masks):
                    action = exit_act

            return action, {}

        # ------------------------------------------------------------
        # 2. cross_2（建玉が無い時のみ適用）
        # ------------------------------------------------------------
        if cross_2 in (SignalSign.POSITIVE, SignalSign.NEGATIVE):

            if position == PositionType.NONE:
                action = self.entry_action(cross_2, strength, fluctuation, masks)

            return action, {}

        # ------------------------------------------------------------
        # 3. ロスカット・利確（建玉がある場合のみ意味がある）
        # ------------------------------------------------------------
        # ロスカットあるいは利確判定用
        exit_flags = (losscut_1, losscut_2, takeprofit_1)
        if any(exit_flags):

            exit_act = self.opposite_action(position)
            if exit_act is not None and self.can_execute(exit_act, masks):
                action = exit_act

        return action, {}

    def updateObs(self, list_obs_label):
        self.list_obs_label = list_obs_label
        self.idx_cross_1 = self.list_obs_label.index("クロスS1")
        self.idx_cross_2 = self.list_obs_label.index("クロスS2")
        self.idx_strength = self.list_obs_label.index("クロ強")
        self.idx_fluc = self.list_obs_label.index("乱高下")
        self.idx_losscut_1 = self.list_obs_label.index("ロス1")
        self.idx_losscut_2 = self.list_obs_label.index("ロス2")
        self.idx_takeprofit_1 = self.list_obs_label.index("利確1")
        self.idx_position = self.list_obs_label.index("建玉")
