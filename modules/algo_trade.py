from structs.app_enum import SignalSign, ActionType


class AlgoTrade:
    """
    強化学習モデルの代わりに、自作のアルゴリズムで取引するクラス
    """
    def __init__(self, list_obs: list):
        self.list_obs = list_obs

    def predict(self, obs, action_masks) -> tuple[int, dict]:
        idx_cross = self.list_obs.index("クロスS")
        # ---------------------------------------------------------------------
        # Signal MAD: MAΔ の符号反転シグナル
        signal_mad = SignalSign(int(obs[idx_cross]))
        if signal_mad == SignalSign.ZERO:
            action = ActionType.HOLD.value
        elif signal_mad == SignalSign.POSITIVE:
            if action_masks[ActionType.BUY.value]:
                action = ActionType.BUY.value
            else:
                action = ActionType.HOLD.value
        elif signal_mad == SignalSign.NEGATIVE:
            if action_masks[ActionType.SELL.value]:
                action = ActionType.SELL.value
            else:
                action = ActionType.HOLD.value
        else:
            raise TypeError(f"Unknown SignalSign: {signal_mad}")

        return action, {}
