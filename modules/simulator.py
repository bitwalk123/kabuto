import pandas as pd

from modules.psar import RealtimePSAR


class TradeSimulator:
    def __init__(self, df: pd.DataFrame, dict_conf: dict):
        self.df = df
        self.dict_conf = dict_conf
        self.psar = RealtimePSAR()

    def run(self):
        # 移動メディアンの算出
        self.df["MMPrice"] = self.df["Price"].rolling(
            self.dict_conf["moving median"],
            min_periods=1
        ).median()

        # Parabolic SAR の算出
        for t in self.df.index:
            p = self.df.at[t, "MMPrice"]
            self.psar.add(p)
            self.df.at[t, "Trend"] = self.psar.obj.trend
            self.df.at[t, "PSAR"] = self.psar.obj.psar
