import pandas as pd

from funcs.conv import min_max_scale
from modules.position_mannager import PositionManager
from modules.psar import RealtimePSAR
from structs.posman import PositionType


class TradeSimulator:
    def __init__(self, ticker: str, df: pd.DataFrame, dict_conf: dict):
        self.ticker = ticker
        self.df = df
        self.dict_conf = dict_conf
        af = dict_conf["AF"]
        af_init = af
        af_step = af
        af_max = af * 100
        rolling_n = dict_conf["rolling N"]
        self.psar = RealtimePSAR(
            af_init, af_step, af_max, rolling_n
        )
        self.posman = PositionManager()
        self.posman.initPosition([ticker])
        self.trend = 0

    def run(self) -> int:
        # 銘柄コード間で比較ができるように、株価を [0, 1] にスケーリング
        self.df["MinMaxPrice"] = min_max_scale(self.df["Price"])
        # 移動メディアンの算出
        self.df["MMPrice"] = self.df["MinMaxPrice"].rolling(
            self.dict_conf["moving median"],
            min_periods=1
        ).median()

        # Parabolic SAR の算出
        for t in self.df.index:
            ts = self.df.at[t, "Time"]
            price = self.df.at[t, "MinMaxPrice"]
            p = self.df.at[t, "MMPrice"]
            self.psar.add(p)

            trend_new = self.psar.obj.trend
            epupd = self.psar.obj.epupd

            self.df.at[t, "Trend"] = trend_new
            self.df.at[t, "EPupd"] = epupd
            self.df.at[t, "PSAR"] = self.psar.obj.psar

            # トレンド反転チェック
            if self.trend != trend_new:
                if self.trend != 0:
                    self.posman.closePosition(self.ticker, ts, price)
                self.trend = trend_new

            if epupd == 1:
                if 0 < self.trend:
                    self.posman.openPosition(self.ticker, ts, price, PositionType.BUY)
                elif self.trend < 0:
                    self.posman.openPosition(self.ticker, ts, price, PositionType.SELL)

        ts = self.df.at[t, "Time"]
        price = self.df.at[t, "MinMaxPrice"]
        self.posman.closePosition(self.ticker, ts, price)
        df_result = self.posman.getTransactionResult()

        return int(df_result["損益"].sum())
