from structs.app_enum import SignalSign, ActionType, PositionType


class AlgoTrade:
    """
    強化学習モデルの代わりに、自作のアルゴリズムで取引するクラス
    """

    def __init__(self, list_obs: list):
        self.list_obs = list_obs

    def predict(self, obs, action_masks) -> tuple[int, dict]:
        # Signal MAD 1: MAΔ の符号反転シグナル 1
        idx_signal_mad_1 = self.list_obs.index("クロスS1")
        signal_mad_1 = SignalSign(int(obs[idx_signal_mad_1]))
        # Signal MAD 1: MAΔ の符号反転シグナル 2
        idx_signal_mad_2 = self.list_obs.index("クロスS2")
        signal_mad_2 = SignalSign(int(obs[idx_signal_mad_2]))
        # Low Volatility Flag - ボラティリティがしきい値より低ければ立つフラグ
        idx_flag_vola_low = self.list_obs.index("低ボラ")
        flag_vola_low = int(obs[idx_flag_vola_low])
        # ポジション（建玉）
        idx_position = self.list_obs.index("建玉")
        position = PositionType(int(obs[idx_position]))
        # ---------------------------------------------------------------------
        # MAΔ の符号反転シグナル 1 の処理
        if signal_mad_1 == SignalSign.ZERO:
            action = ActionType.HOLD.value
        elif signal_mad_1 == SignalSign.POSITIVE and action_masks[ActionType.BUY.value]:
            action = ActionType.BUY.value
        elif signal_mad_1 == SignalSign.NEGATIVE and action_masks[ActionType.SELL.value]:
            action = ActionType.SELL.value
        else:
            action = ActionType.HOLD.value
        # ---------------------------------------------------------------------
        # MAΔ の符号反転シグナル 2（ディレイ）の処理（反対売買用）
        if signal_mad_2 == SignalSign.POSITIVE and action_masks[ActionType.BUY.value]:
            action = ActionType.BUY.value
        elif signal_mad_2 == SignalSign.NEGATIVE and action_masks[ActionType.SELL.value]:
            action = ActionType.SELL.value
        # ---------------------------------------------------------------------
        # 低ボラ時のエントリ強制禁止処理
        if flag_vola_low and position == PositionType.NONE:
            action = ActionType.HOLD.value

        return action, {}
