from structs.app_enum import SignalSign, ActionType, PositionType


class AlgoTrade:
    """
    強化学習モデルの代わりに、自作のアルゴリズムで取引するクラス
    """

    def __init__(self, list_obs: list):
        self.list_obs = list_obs

    def predict(self, obs, masks) -> tuple[int, dict]:
        # クロスシグナル 1
        idx_cross_1 = self.list_obs.index("クロスS1")
        cross_1 = SignalSign(int(obs[idx_cross_1]))
        # クロスシグナル 2
        idx_cross_2 = self.list_obs.index("クロスS2")
        cross_2 = SignalSign(int(obs[idx_cross_2]))
        # クロスシグナル強度
        idx_strength = self.list_obs.index("クロ強")
        strength = int(obs[idx_strength])
        # ポジション（建玉）
        idx_position = self.list_obs.index("建玉")
        position = PositionType(int(obs[idx_position]))
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # クロスシグナルの処理
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        """
        cross_1 は建玉の有無にかかわらず適用を試みる
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
                if masks[ActionType.SELL.value] == 1:
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
            cross_2 は建玉が無い時のみに適用を試みる
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
                    if masks[ActionType.SELL.value] == 1:
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
                action = ActionType.HOLD.value

        return action, {}
