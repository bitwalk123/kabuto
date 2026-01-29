import numpy as np

from modules.feature_provider import FeatureProvider

FEATURES = [
    ("クロスS1", "getCrossSignal1"),  # 移動平均のクロスシグナル 1 [-1, 0, 1]
    ("ロス1", "getLosscut1"),  # ロスカット 1 [0, 1]
    ("ロス2", "getLosscut2"),  # ロスカット 2 [0, 1]
    ("利確1", "doesTakeProfit"),  # 利確 1 [0, 1]
    ("建玉", "getPositionValue"),  # ポジション情報 [-1, 0, 1]
]

TECHNICALS = {
    "ts": "getTimestamp",  # タイムスタンプ
    "price": "getPrice",  # 株価
    "ma1": "getMA1",  # 移動平均線 MA1
    "ma2": "getMA2",  # 移動平均線 MA2
    "profit": "getProfit",  # 含損益
    "profit_max": "getProfitMax",  # 最大含み損益
    "drawdown": "getDrawDown",  # ドローダウン
    "dd_ratio": "getDDRatio",  # ドローダウン比率
    "n_minus": "getCounterMinus",  # 含み益が負の時のカウンタ
}


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

    def getObs(self):
        p = self.provider

        # 観測値（特徴量）用リスト
        list_feature = [getattr(p, fn)() for _, fn in FEATURES]

        # プロット用データ
        dict_technicals = {
            key: getattr(p, name)() if callable(getattr(p, name)) else getattr(p, name)
            for key, name in TECHNICALS.items()
        }

        return np.array(list_feature, dtype=np.float32), dict_technicals

    @staticmethod
    def getObsList() -> list:
        return [name for name, _ in FEATURES]

    def getObsReset(self) -> np.ndarray:
        obs, _ = self.getObs()
        return obs
