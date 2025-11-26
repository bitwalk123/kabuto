import numpy as np

from modules.feature_provider import FeatureProvider
from structs.app_enum import PositionType


class ObservationManager:
    def __init__(self, provider: FeatureProvider):
        # 特徴量プロバイダ
        self.provider = provider

        # 取引用係数
        self.price_tick = 1.0  # 呼び値
        self.unit = 100  # 最小取引単位（出来高）

        self.signal_mad_pre = 0.0
        self.emad_signal_pre = 0.0
        self.position_reverse = False

        # ---------------------------------------------------------------------
        # 調整用係数
        self.divisor_hold = 250.0  # 建玉なしの HOLD カウンタ用除数
        self.divisor_hold_position = 10_000.0  # 建玉ありの HOLD カウンタ用
        self.divisor_ma_diff = 50.0  # 移動平均差用
        self.multiplier_price = 20.0  # 株価用
        self.multiplier_vwap = 40.0  # VWAP用
        self.divisor_msd = 25.0  # 移動標準偏差用
        # ---------------------------------------------------------------------

        """
        観測量（特徴量）数の取得
        観測量の数 (self.n_feature) は、評価によって頻繁に変動するので、
        コンストラクタでダミー（空）処理を実行して数を自律的に把握できるようにする。
        """
        self.n_feature = len(self.getObs())
        self.clear()  # ダミー処理を実行したのでリセット

    def clear(self):
        self.provider.clear()

    def getObs(self) -> np.ndarray:
        # 観測値（特徴量）用リスト
        list_feature = list()
        # ---------------------------------------------------------------------
        # 0. Position Reverse 反対売買許可フラグ（リセットされる前の状態を渡す）
        if self.position_reverse:
            signal_reverse = 1
        else:
            signal_reverse = 0
        list_feature.append(signal_reverse)
        # ---------------------------------------------------------------------
        # 1. MAΔS+（MAΔ の符号反転シグナル、反対売買、ボラティリティによるエントリ制御）
        signal_mad = self.provider.getMADSignal()

        if self.provider.position == PositionType.NONE:
            # ポジション無し
            if self.position_reverse:
                """
                self.position_reverse で反対売買が許可されていて建玉が無い場合は、
                反対売買をして、フラグをリセットする。
                """
                signal_mad = self.signal_mad_pre
                self.signal_mad_pre = 0
                self.position_reverse = False

            if signal_mad != 0 and self.provider.isLowVolatility():
                """
                ボラティリティが小さい時はエントリを禁止
                """
                signal_mad = 0
        else:
            # ポジション有り
            if signal_mad != 0:
                """
                mad_signal が立った時（クロス時）に建玉を持っている場合は、
                次のステップでも同じフラグを立てて反対売買を許容する。
                """
                self.signal_mad_pre = signal_mad
                self.position_reverse = True

        list_feature.append(signal_mad)
        # ---------------------------------------------------------------------
        # 2. Low Volatility Flag - ボラティリティがしきい値より低ければフラグを立てる
        if self.provider.isLowVolatility():
            vol_low = 1
        else:
            vol_low = 0
        list_feature.append(vol_low)
        # ---------------------------------------------------------------------
        # 3. ポジション情報
        if self.provider.position == PositionType.NONE:
            value_position = 0
        elif self.provider.position == PositionType.LONG:
            value_position = 1
        elif self.provider.position == PositionType.SHORT:
            value_position = -1
        else:
            raise TypeError(f"Unknown PositionType: {self.provider.position}")
        list_feature.append(value_position)
        # ---------------------------------------------------------------------
        # 4 含損益
        list_feature.append(self.provider.get_profit())
        # ---------------------------------------------------------------------
        # 5. 含損益M（含み損益最大）
        list_feature.append(self.provider.profit_max)
        # ---------------------------------------------------------------------
        # 配列にして観測値を返す
        return np.array(list_feature, dtype=np.float32)

    @staticmethod
    def getObsList() -> list:
        return [
            "反対売買",
            "クロスS",
            "低ボラ",
            "ポジション",
            "含損益",
            "含損益M",
        ]

    def getObsReset(self) -> np.ndarray:
        obs = self.getObs()  # 引数無しで呼んだ場合、ダミーの観測値が返る
        self.clear()
        return obs
