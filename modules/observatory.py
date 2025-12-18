from collections import deque

import numpy as np

from modules.feature_provider import FeatureProvider


class ObservationManager:
    def __init__(self, provider: FeatureProvider):
        # 特徴量プロバイダ
        self.provider = provider

        # 取引用係数
        self.price_tick = 1.0  # 呼び値
        self.unit = 100  # 最小取引単位（出来高）

        # キュー
        self.deque_signal_mad = deque(maxlen=2)
        # ---------------------------------------------------------------------
        # 調整用係数
        # ---------------------------------------------------------------------

        """
        観測量（特徴量）数の取得
        観測量の数 (self.n_feature) は、評価によって頻繁に変動するので、
        コンストラクタでダミー（空）処理を実行して数を自律的に把握できるようにする。
        """
        self.n_feature = len(self.getObs())

    def getObs(self) -> np.ndarray:
        # 観測値（特徴量）用リスト
        list_feature = list()
        # ---------------------------------------------------------------------
        # 0. MA1（移動平均 1）
        ma_1 = self.provider.getMA1()
        list_feature.append(ma_1)
        # ---------------------------------------------------------------------
        # 1. MA2（移動平均 2）
        ma_2 = self.provider.getMA2()
        list_feature.append(ma_2)
        # ---------------------------------------------------------------------
        # 2. MAΔS1（MAΔ の符号反転シグナル）
        signal_mad_sign = self.provider.getMADSignal()
        signal_mad_1 = float(signal_mad_sign.value)  # 数値化
        self.deque_signal_mad.append(signal_mad_1)
        list_feature.append(signal_mad_1)
        # ---------------------------------------------------------------------
        # 3. MAΔS2（MAΔ の符号反転シグナル）ひとつ前
        # ---------------------------------------------------------------------
        if len(self.deque_signal_mad) > 1:
            signal_mad_2 = self.deque_signal_mad[-2]
        else:
            signal_mad_2 = 0.0
        list_feature.append(signal_mad_2)
        # ---------------------------------------------------------------------
        # 4. Low Volatility Flag - ボラティリティがしきい値より低ければフラグを立てる
        if self.provider.isLowVolatility():
            flag_vola_low = 1
        else:
            flag_vola_low = 0
        list_feature.append(flag_vola_low)
        # ---------------------------------------------------------------------
        # 5. 移動範囲
        mr = self.provider.mr
        list_feature.append(mr)
        # ---------------------------------------------------------------------
        # 6. ポジション情報
        value_position = float(self.provider.position.value)  # 数値化
        list_feature.append(value_position)
        # ---------------------------------------------------------------------
        # 7. 含損益
        profit_unrealized = self.provider.get_profit()
        list_feature.append(profit_unrealized)
        # ---------------------------------------------------------------------
        # 8. 含損益M（含み損益最大）
        profit_unrealized_max = self.provider.profit_max
        list_feature.append(profit_unrealized_max)
        # ---------------------------------------------------------------------
        # 9. ロスカット・プラグ
        """
        flag_losscut = self.provider.doesLossCut()
        """
        flag_losscut = False
        list_feature.append(flag_losscut)
        # ---------------------------------------------------------------------
        # 10. 利確プラグ
        """
        if self.provider.doesTakeProfit():
            flag_take_profit = 1
        else:
            flag_take_profit = 0
        """
        flag_take_profit = 0
        list_feature.append(flag_take_profit)
        # =====================================================================
        # 配列にして観測値を返す
        return np.array(list_feature, dtype=np.float32)

    @staticmethod
    def getObsList() -> list:
        return [
            "MA1",
            "MA2",
            "クロスS1",
            "クロスS2",
            "低ボラ",
            "MR",
            "建玉",
            "含損益",
            "損益M",
            "ロス",
            "利確",
        ]

    def getObsReset(self) -> np.ndarray:
        obs = self.getObs()  # 引数無しで呼んだ場合、ダミーの観測値が返る
        return obs
