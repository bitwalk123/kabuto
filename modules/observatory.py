from modules.env_data import EnvData
from modules.technical import (
    MovingAverage,
    PurePursuitFollower,
    VWAP,
)


class ObservationManager:
    def __init__(self, s: EnvData):
        # 特徴量プロバイダ
        self.s = s
        # 特徴量インスタンス
        # self.ma_1 = MovingAverage(window_size=self.s.PERIOD_MA_1)
        self.ppf = PurePursuitFollower()
        # self.ma_1 = HMA(window_size=self.s.PERIOD_MA_1)
        self.ma_2 = MovingAverage(window_size=self.s.PERIOD_MA_2)
        # self.ma_2 = HMA(window_size=self.s.PERIOD_MA_2)
        #self.mom = Momentum(window_size=self.s.PERIOD_MOM)
        self.vwap = VWAP()

    def update(self, ts: float, price: float, volume: float) -> dict:
        value_ppf, value_mom = self.ppf.update(price)
        value_ma_2 = self.ma_2.update(price)
        value_vwap = self.vwap.update(price, volume)

        return {
            "Time": ts,
            "Price": price,
            "MA1": value_ppf,
            "MA2": value_ma_2,
            "DiffMA": (value_ppf - value_ma_2) / value_ma_2,
            "VWAP": value_vwap,
            "DiffVWAP": (value_ppf - value_vwap) / value_vwap,
            "RSI": 0,
            "Momentum": value_mom,
        }
