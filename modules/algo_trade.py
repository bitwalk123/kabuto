from structs.app_enum import SignalSign, ActionType, PositionType


class AlgoTrade:
    """
    強化学習モデルの代わりに、自作のアルゴリズムで取引するクラス
    """

    def __init__(self, list_obs_label: list):
        self.list_obs_label = list_obs_label

    def getListObs(self) -> list:
        return self.list_obs_label

    def predict(self, obs, masks) -> tuple[int, dict]:
        # 0. クロスシグナル 1
        idx_cross_1 = self.list_obs_label.index("クロスS1")
        cross_1 = SignalSign(int(obs[idx_cross_1]))
        # 1. クロスシグナル 2
        idx_cross_2 = self.list_obs_label.index("クロスS2")
        cross_2 = SignalSign(int(obs[idx_cross_2]))
        # 2. クロスシグナル強度
        idx_strength = self.list_obs_label.index("クロ強")
        strength = int(obs[idx_strength])
        # 3. ロスカット 1
        idx_losscut_1 = self.list_obs_label.index("ロス1")
        losscut_1 = int(obs[idx_losscut_1])
        # 4. ポジション（建玉）
        idx_position = self.list_obs_label.index("建玉")
        position = PositionType(int(obs[idx_position]))

        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # シグナルの処理
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        """
        クロスシグナル cross_1 は建玉の有無にかかわらず適用を試みる
        """
        if cross_1 == SignalSign.POSITIVE:
            if position == PositionType.NONE:
                # エントリ
                if strength and masks[ActionType.BUY.value] == 1:
                    action = ActionType.BUY.value
                else:
                    action = ActionType.HOLD.value
            elif position == PositionType.LONG:
                if masks[ActionType.SELL.value] == 1:
                    action = ActionType.SELL.value
                else:
                    action = ActionType.HOLD.value
            elif position == PositionType.SHORT:
                if masks[ActionType.BUY.value] == 1:
                    action = ActionType.BUY.value
                else:
                    action = ActionType.HOLD.value
            else:
                raise TypeError(f"Unknown PositionType: {position}")
        elif cross_1 == SignalSign.NEGATIVE:
            if position == PositionType.NONE:
                # エントリ
                if strength and masks[ActionType.SELL.value] == 1:
                    action = ActionType.SELL.value
                else:
                    action = ActionType.HOLD.value
            elif position == PositionType.LONG:
                if masks[ActionType.SELL.value] == 1:
                    action = ActionType.SELL.value
                else:
                    action = ActionType.HOLD.value
            elif position == PositionType.SHORT:
                if masks[ActionType.BUY.value] == 1:
                    action = ActionType.BUY.value
                else:
                    action = ActionType.HOLD.value
            else:
                raise TypeError(f"Unknown PositionType: {position}")
        else:
            """
            クロスシグナル cross_2 は建玉が無い時のみに適用を試みる
            """
            if cross_2 == SignalSign.POSITIVE:
                if position == PositionType.NONE:
                    # エントリ
                    if strength and masks[ActionType.BUY.value] == 1:
                        action = ActionType.BUY.value
                    else:
                        action = ActionType.HOLD.value
                elif position == PositionType.LONG:
                    action = ActionType.HOLD.value
                elif position == PositionType.SHORT:
                    action = ActionType.HOLD.value
                else:
                    raise TypeError(f"Unknown PositionType: {position}")
            elif cross_2 == SignalSign.NEGATIVE:
                if position == PositionType.NONE:
                    # エントリ
                    if strength and masks[ActionType.SELL.value] == 1:
                        action = ActionType.SELL.value
                    else:
                        action = ActionType.HOLD.value
                elif position == PositionType.LONG:
                    action = ActionType.HOLD.value
                elif position == PositionType.SHORT:
                    action = ActionType.HOLD.value
                else:
                    raise TypeError(f"Unknown PositionType: {position}")
            else:
                """
                ログカット 1（単純ロスカット）
                """
                if losscut_1:
                    if position == PositionType.LONG:
                        if masks[ActionType.SELL.value] == 1:
                            action = ActionType.SELL.value
                        else:
                            action = ActionType.HOLD.value
                    elif position == PositionType.SHORT:
                        if masks[ActionType.BUY.value] == 1:
                            action = ActionType.BUY.value
                        else:
                            action = ActionType.HOLD.value
                    else:
                        action = ActionType.HOLD.value
                else:
                    action = ActionType.HOLD.value

        return action, {}
