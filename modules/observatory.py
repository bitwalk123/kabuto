from modules.env_data import EnvData
from modules.technical import (
    Momentum,
    MovingAverage,
    VWAP,
)


class ObservationManager:
    def __init__(self, s: EnvData):
        # 特徴量プロバイダ
        self.s = s
        # 特徴量インスタンス
        self.ma_1 = MovingAverage(window_size=self.s.PERIOD_MA_1)
        # self.ma_1 = PurePersuitFollower()
        self.ma_2 = MovingAverage(window_size=self.s.PERIOD_MA_2)
        self.mom = Momentum(window_size=self.s.PERIOD_MOM)
        self.vwap = VWAP()

    def update(self, ts: float, price: float, volume: float) -> dict:
        """
        self.ts = row["Time"]
        self.price = row["Price"]
        self.ma1 = row["MA1"]
        self.ma2 = row["MA2"]
        self.diff_ma = row["DiffMA"]
        self.vwap = row["VWAP"]
        self.diff_vwap = row["DiffVWAP"]
        self.rsi = row["RSI"]
        self.mom = row["Momentum"]
        """
        value_ma_1 = self.ma_1.update(price)
        value_ma_2 = self.ma_2.update(price)
        value_vwap = self.vwap.update(price, volume)

        return {
            "Time": ts,
            "Price": price,
            "MA1": value_ma_1,
            "MA2": value_ma_2,
            "DiffMA": 0,
            "VWAP": value_vwap,
            "DiffVWAP": (value_ma_1 - value_vwap) / value_vwap * 100.,
            "RSI": 0,
            "Momentum": self.mom.update(value_ma_1),
        }
