from structs.app_enum import ActionType, PositionType


class AlgoTrade:
    """
    強化学習モデルの代わりに、自作のアルゴリズムで取引する疑似モデルのクラス
    """

    def __init__(self):
        self.list_obs_label = None
        self.idx_cross_1 = None
        self.idx_losscut_1 = None
        self.idx_takeprofit_1 = None
        self.idx_position = None

    def getListObs(self) -> list:
        return self.list_obs_label

    @staticmethod
    def can_execute(action, masks):
        """
        アクションが行動マスクで禁止されていないかチェック
        :param action:
        :param masks:
        :return:
        """
        return masks[action] == 1

    @staticmethod
    def exit_action(position: PositionType):
        """
        ポジションに応じた返済アクション
        【備考】
        以前は返済アクションにREPAYMENTを用意していたが、
        モデルの学習に合わないので削除した。→ 疑似モデルでも維持
        :param position:
        :return:
        """
        if position == PositionType.LONG:
            return ActionType.SELL.value
        if position == PositionType.SHORT:
            return ActionType.BUY.value
        return None

    def predict(self, obs, masks) -> tuple[int, dict]:
        # --- 観測値の取り出し ---
        # 1. クロスシグナル 1 [-1, 0, 1]
        cross_1 = int(obs[self.idx_cross_1])
        # 2. ロスカット 1 [0, 1]
        losscut_1 = int(obs[self.idx_losscut_1])
        # 3. ロスカット 2 [0, 1]
        losscut_2 = int(obs[self.idx_losscut_2])
        # 4. 利確 1 [0, 1]
        takeprofit_1 = int(obs[self.idx_takeprofit_1])
        # 5. ポジション（建玉） [SHORT, NONE, LONG]
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

        # 3. デフォルトは HOLD
        return ActionType.HOLD.value, {}

    def updateObs(self, list_obs_label):
        """
        疑似ロジックでは、観測値にラベルを付けておかないと、コーディングする側が間違える！
        :param list_obs_label:
        :return:
        """
        self.list_obs_label = list_obs_label
        self.idx_cross_1 = self.list_obs_label.index("クロスS1")
        self.idx_losscut_1 = self.list_obs_label.index("ロス1")
        self.idx_losscut_2 = self.list_obs_label.index("ロス2")
        self.idx_takeprofit_1 = self.list_obs_label.index("利確1")
        self.idx_position = self.list_obs_label.index("建玉")
