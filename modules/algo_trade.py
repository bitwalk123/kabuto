from structs.app_enum import SignalSign, ActionType, PositionType


class AlgoTrade:
    """
    強化学習モデルの代わりに、自作のアルゴリズムで取引するクラス
    """

    def __init__(self, list_obs: list):
        self.list_obs = list_obs

    def predict(self, obs, masks) -> tuple[int, dict]:
        # クロスシグナル
        idx_signal_cross = self.list_obs.index("クロス")
        cross = SignalSign(int(obs[idx_signal_cross]))
        # クロスシグナル強度
        idx_signal_strength = self.list_obs.index("クロ強")
        signal_strength = int(obs[idx_signal_strength])
        # ポジション（建玉）
        idx_position = self.list_obs.index("建玉")
        position = PositionType(int(obs[idx_position]))
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # クロスシグナルの処理
        if position == PositionType.LONG:
            if cross != SignalSign.ZERO and masks[ActionType.SELL.value] == 1:
                action = ActionType.SELL.value
            else:
                action = ActionType.HOLD.value
        elif position == PositionType.SHORT:
            if cross != SignalSign.ZERO and masks[ActionType.BUY.value] == 1:
                action = ActionType.BUY.value
            else:
                action = ActionType.HOLD.value
        else:
            if cross == SignalSign.POSITIVE and masks[ActionType.BUY.value] == 1:
                action = ActionType.BUY.value
            elif cross == SignalSign.NEGATIVE and masks[ActionType.SELL.value] == 1:
                action = ActionType.SELL.value
            else:
                action = ActionType.HOLD.value

        return action, {}
