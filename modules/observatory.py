import numpy as np

from modules.features import FeatureProvider
from structs.app_enum import PositionType


class ObservationManager:
    def __init__(self, provider: FeatureProvider):
        # 特徴量プロバイダ
        self.provider = provider
        # 調整用係数
        self.price_tick = 1.0  # 呼び値
        self.unit = 100  # 最小取引単位（出来高）
        self.divisor_hold = 10_000.  # 建玉保持カウンタ用
        self.divisor_ma_diff = 20.0  # 移動平均差用
        self.multiplier_price = 20.0  # 株価用
        self.multiplier_vwap = 40.0  # VWAP用
        self.divisor_msd = 25.0  # 移動標準偏差用
        """
        観測量（特徴量）数の取得
        観測量の数 (self.n_feature) は、評価によって頻繁に変動するので、
        コンストラクタでダミー（空）を実行して数を自律的に把握できるようにする。
        """
        self.n_feature = len(self.getObs())
        self.clear()  # ダミーを実行したのでリセット

    def clear(self):
        self.provider.clear()

    def func_ma_scaling(self, ma: float) -> float:
        if self.provider.price_open == 0.0:
            ma_ratio = 0.0
        else:
            ma_ratio = (ma / self.provider.price_open - 1.0) * self.multiplier_price
        return ma_ratio

    def func_ma_diff_scaling(self, ma_diff: float) -> float:
        if self.provider.price_open == 0.0:
            ma_diff_scaled = 0.0
        else:
            ma_diff_scaled = ma_diff / self.price_tick / self.divisor_ma_diff
        return np.tanh(ma_diff_scaled)

    def getObs(
            self,
            pl: float = 0,  # 含み損益
            pl_max: float = 0,  # 含み損益（最大値）
            position: PositionType = PositionType.NONE  # ポジション
    ) -> np.ndarray:
        # 観測値（特徴量）用リスト
        list_feature = list()
        # ---------------------------------------------------------------------
        # 1. 株価比率
        # ---------------------------------------------------------------------
        price_ratio = self.provider.getPriceRatio()
        price_ratio = (price_ratio - 1.0) * self.multiplier_price
        list_feature.append(price_ratio)
        # 移動平均
        ma_060 = self.provider.getMA(60)
        ma_300 = self.provider.getMA(300)
        # ---------------------------------------------------------------------
        # 2. MAΔ: 移動平均の差分 MA60 - MA300
        # ---------------------------------------------------------------------
        # 移動平均差の算出
        ma_diff = ma_060 - ma_300
        list_feature.append(self.func_ma_diff_scaling(ma_diff))
        # ---------------------------------------------------------------------
        # 3. VWAP 乖離率 (deviation rate = dr)
        # ---------------------------------------------------------------------
        vwap_dr = self.provider.getVWAPdr()
        vwap_dr_scaled = np.tanh(vwap_dr * self.multiplier_vwap)
        list_feature.append(vwap_dr_scaled)
        # ---------------------------------------------------------------------
        # 4. 移動標準偏差
        # ---------------------------------------------------------------------
        msd = self.provider.getMSD(60)
        msd_scaled = np.tanh(msd / self.price_tick / self.divisor_msd)
        list_feature.append(msd_scaled)
        # ---------------------------------------------------------------------
        # 5. 含み損益
        # ---------------------------------------------------------------------
        list_feature.append(pl)
        # ---------------------------------------------------------------------
        # 6. 含み損益（最大）
        # ---------------------------------------------------------------------
        list_feature.append(pl_max)
        # ---------------------------------------------------------------------
        # 7. HOLD 継続カウンタ 2（建玉なし）
        # ---------------------------------------------------------------------
        list_feature.append(np.tanh(self.provider.n_hold / self.provider.n_hold_divisor))
        # ---------------------------------------------------------------------
        # 8. HOLD 継続カウンタ 2（建玉あり）
        # ---------------------------------------------------------------------
        list_feature.append(self.provider.n_hold_position / self.divisor_hold)
        # ---------------------------------------------------------------------
        # 9. 取引回数
        # ---------------------------------------------------------------------
        ratio_trade_count = self.provider.n_trade / self.provider.n_trade_max
        list_feature.append(np.tanh(ratio_trade_count))
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # 一旦、配列に変換
        arr_feature = np.array(list_feature, dtype=np.float32)
        # ---------------------------------------------------------------------
        # ポジション情報
        # 10., 11., 12. PositionType → one-hot (3) ［単位行列へ変換］
        # ---------------------------------------------------------------------
        pos_onehot = np.eye(len(PositionType))[position.value].astype(np.float32)
        # arr_feature と pos_onehot を単純結合
        return np.concatenate([arr_feature, pos_onehot])

    def getObsReset(self) -> np.ndarray:
        obs = self.getObs()
        self.clear()
        return obs
