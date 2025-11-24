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

        self.mad_signal_pre = 0.0
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

    def getObs(
            self,
            pl: float = 0,  # 含み損益
            pl_max: float = 0,  # 含み損益（最大値）
            position: PositionType = PositionType.NONE  # ポジション
    ) -> np.ndarray:
        # 観測値（特徴量）用リスト
        list_feature = list()
        """
        # ---------------------------------------------------------------------
        # ?. 株価比
        price_ratio = self.provider.getPriceRatio()
        price_ratio = (price_ratio - 1.0) * self.multiplier_price
        list_feature.append(price_ratio)
        # ---------------------------------------------------------------------
        # ?. MAΔ（異なる２つの移動平均の差分）
        if self.provider.price_open == 0.0:
            mad_scaled = 0.0
        else:
            mad = self.provider.getMAD()
            mad_scaled = np.tanh(mad / self.price_tick / self.divisor_ma_diff)
        list_feature.append(mad_scaled)
        """
        # ---------------------------------------------------------------------
        # 1. MAΔS+（MAΔ の符号反転シグナル、反対売買、ボラティリティによるエントリ制御）
        mad_signal = self.provider.getMADSignal()
        if mad_signal != 0 and position != PositionType.NONE:
            """
            mad_signal が立った時（クロス時）に建玉を持っている場合は、
            次のステップでも同じフラグを立てて反対売買を許容する。
            """
            self.mad_signal_pre = mad_signal
            self.position_reverse = True
        elif self.position_reverse and position == PositionType.NONE:
            """
            self.position_reverse で反対売買が許可されていて建玉が無い場合は、
            反対売買をして、フラグをリセットする。
            """
            mad_signal = self.mad_signal_pre
            self.mad_signal_pre = 0.0
            self.position_reverse = False

        list_feature.append(mad_signal)
        # ---------------------------------------------------------------------
        """
        # ?. VWAPΔ（VWAP 乖離率, deviation rate = dr）
        vwap_dr = self.provider.getVWAPdr()
        vwap_dr_scaled = np.tanh(vwap_dr * self.multiplier_vwap)
        list_feature.append(vwap_dr_scaled)
        # ---------------------------------------------------------------------
        # ?. Mσ（移動標準偏差, Moving σ）
        msd = self.provider.getMSD()
        msd_scaled = np.tanh(msd / self.price_tick / self.divisor_msd)
        # list_feature.append(msd)
        list_feature.append(msd_scaled)
        # ---------------------------------------------------------------------
        # ?. 含損益
        list_feature.append(pl)
        # ---------------------------------------------------------------------
        # ?. 含損益M（含み損益最大）
        list_feature.append(pl_max)
        # ---------------------------------------------------------------------
        # ?. HOLD1（継続カウンタ 1, 建玉なし）
        hold_1_scaled = np.tanh(self.provider.n_hold / self.divisor_hold)
        list_feature.append(hold_1_scaled)
        # ---------------------------------------------------------------------
        # ?. HOLD2（継続カウンタ 2, 建玉あり）
        hold_2_scaled = self.provider.n_hold_position / self.divisor_hold_position
        list_feature.append(hold_2_scaled)
        # ---------------------------------------------------------------------
        # ?. TRADE（取引回数）
        ratio_trade_count = self.provider.n_trade / self.provider.n_trade_max
        list_feature.append(np.tanh(ratio_trade_count))
        # ---------------------------------------------------------------------

        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # 一旦、配列に変換
        arr_feature = np.array(list_feature, dtype=np.float32)

        # ---------------------------------------------------------------------
        # ポジション情報
        # >. NONE, ?. LONG, ?. SHORT
        # PositionType → one-hot (3) ［単位行列へ変換］
        pos_onehot = np.eye(len(PositionType))[position.value].astype(np.float32)
        # ---------------------------------------------------------------------

        # arr_feature と pos_onehot を単純結合
        return np.concatenate([arr_feature, pos_onehot])
        """
        return np.array(list_feature, dtype=np.float32)

    @staticmethod
    def getObsList() -> list:
        return [
            #"株価比",
            #"MAΔ",
            "MAΔS",
            #"VWAPΔ",
            #"Mσ",
            #"含損益",
            #"含損益M",
            #"HOLD1",
            #"HOLD2",
            #"TRADE",
            #"NONE",
            #"LONG",
            #"SHORT"
        ]

    def getObsReset(self) -> np.ndarray:
        obs = self.getObs()  # 引数無しで呼んだ場合、ダミーの観測値が返る
        self.clear()
        return obs
