import numpy as np
import pandas as pd


class ParabolicSAR:
    def __init__(self, af_init: float = 0.02, af_step: float = 0.02, af_max: float = 0.2):
        self.af_init = af_init
        self.af_step = af_step
        self.af_max = af_max

    def calc(self, df: pd.DataFrame):
        length = len(df)

        af = self.af_init

        df["Trend"] = np.nan
        df["EP"] = np.nan
        df["AF"] = np.nan
        df["PSAR"] = df["Close"]

        t = df.index[1]
        trend = 1
        df.at[t, "Trend"] = trend
        df.at[t, "EP"] = df.at[t, "Low"]

        for i in range(2, length):
            t0 = df.index[i - 2]
            t1 = df.index[i - 1]
            t2 = df.index[i]

            # SAR（Stop And Reverse Point） = 前日のSAR ＋ AF ✕（EP － 前日のSAR）
            df.at[t2, "PSAR"] = df.at[t1, "PSAR"] + af * (df.at[t1, "EP"] - df.at[t1, "PSAR"])
            reverse = False

            if trend > 0:
                if df.at[t2, "Low"] < df.at[t2, "PSAR"]:
                    trend = -1
                    reverse = True
                    df.at[t2, "PSAR"] = df.at[t1, "EP"]
                    df.at[t2, "EP"] = df.at[t2, "Low"]
                    af = self.af_init
            else:
                if df.at[t2, "High"] > df.at[t2, "PSAR"]:
                    trend = 1
                    reverse = True
                    df.at[t2, "PSAR"] = df.at[t1, "EP"]
                    df.at[t2, "EP"] = df.at[t2, "High"]
                    af = self.af_init

            if not reverse:
                if trend > 0:
                    if df.at[t2, "High"] > df.at[t1, "EP"]:
                        # EP の更新
                        df.at[t2, "EP"] = df.at[t2, "High"]
                        # AF の更新
                        af = min(af + self.af_step, self.af_max)
                    else:
                        df.at[t2, "EP"]=df.at[t1, "EP"]

                    if df.at[t1, "Low"] < df.at[t2, "PSAR"]:
                        df.at[t2, "PSAR"] = df.at[t1, "Low"]
                    if df.at[t0, "Low"] < df.at[t2, "PSAR"]:
                        df.at[t2, "PSAR"] = df.at[t0, "Low"]
                else:
                    if df.at[t2, "Low"] < df.at[t1, "EP"]:
                        # EP の更新
                        df.at[t2, "EP"] = df.at[t2, "Low"]
                        # AF の更新
                        af = min(af + self.af_step, self.af_max)
                    else:
                        df.at[t2, "EP"] = df.at[t1, "EP"]

                    if df.at[t1, "High"] > df.at[t2, "PSAR"]:
                        df.at[t2, "PSAR"] = df.at[t1, "High"]
                    if df.at[t0, "High"] > df.at[t2, "PSAR"]:
                        df.at[t2, "PSAR"] = df.at[t0, "High"]

            df.at[t2, "Trend"] = trend
            df.at[t2, "AF"] = af

            # mplfinance 用の Bull/Bear
            if trend > 0:
                df.at[t2, "Bull"] = df.at[t2, "PSAR"]
            else:
                df.at[t2, "Bear"] = df.at[t2, "PSAR"]

