import numpy as np

from modules.feature_provider import FeatureProvider


class ObservationManager:
    def __init__(self, provider: FeatureProvider):
        # 特徴量プロバイダ
        self.provider = provider

        """
        観測量（特徴量）数の取得
        観測量の数 (self.n_feature) は、評価によって頻繁に変動するので、
        コンストラクタでダミー（空）処理を実行して数を自律的に把握できるようにする。
        """
        self.n_feature = len(self.getObs()[0])

    def getObs(self) -> tuple[np.ndarray, dict]:
        # プロット用生データ
        dict_technicals = {
            "ts": self.provider.ts,  # タイムsタンプ
            "ma1": self.provider.getMA1(),  # MA1（移動平均 1）
            "ma2": self.provider.getMA2(),  # MA2（移動平均 2）
            "profit": self.provider.get_profit(),  # 含損益
            "profit_max": self.provider.profit_max,  # 最大含み損益
        }
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # 観測値（特徴量）用リスト
        list_feature = list()
        # ---------------------------------------------------------------------
        # 0. 移動平均のクロスシグナル（なし: 0、あり: 1）
        list_feature.append(self.provider.getCrossSignal())
        # ---------------------------------------------------------------------
        # 1. クロスシグナル強度 (なし/弱: 0、強: 1)
        list_feature.append(self.provider.getCrossSignalStrength())
        # ---------------------------------------------------------------------
        # 2. ポジション情報
        list_feature.append(float(self.provider.position.value))
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # 配列にして観測値を返す
        return np.array(list_feature, dtype=np.float32), dict_technicals

    @staticmethod
    def getObsList() -> list:
        return [
            "クロス",
            "クロ強",
            "建玉",
        ]

    def getObsReset(self) -> np.ndarray:
        obs, _ = self.getObs()
        return obs
