from collections import deque

import pandas as pd


def calc_vwap(df: pd.DataFrame):
    """
    ティックデータから VWAP を一括計算
    :param df:
    :return:
    """
    # 差分出来高を算出（最初の行は初期出来高、2 行目以降は増加分）
    df["vol_diff"] = df["Volume"].diff().fillna(df["Volume"])
    # VWAP の計算
    df["cum_pv"] = (df["Price"] * df["vol_diff"]).cumsum()
    df["cum_vol"] = df["vol_diff"].cumsum()
    df["VWAP"] = df["cum_pv"] / df["cum_vol"]
    # VWAP乖離率の計算
    df["VWAP_deviation"] = (df["Price"] - df["VWAP"]) / df["VWAP"]


def calc_ma(df: pd.DataFrame, period_1: int = 60, period_2: int = 600) -> tuple[str, str]:
    """
    移動平均差
    :param df:
    :param period_1:
    :param period_2:
    :return:
    """
    colname1 = f"MA{period_1:03d}"
    colname2 = f"MA{period_2:03d}"
    df[colname1] = df["Price"].rolling(period_1, min_periods=1).mean()
    df[colname2] = df["Price"].rolling(period_2, min_periods=1).mean()
    return colname1, colname2


def calc_miqr(df: pd.DataFrame, period: int = 60) -> str:
    """
    移動IQR
    :param df:
    :param period:
    :return:
    """
    colname = f"MIQR{period:03d}"
    q1 = df["Price"].rolling(window=period, min_periods=1).quantile(0.25)
    q3 = df["Price"].rolling(window=period, min_periods=1).quantile(0.75)

    # IQR 列を作成
    df[colname] = q3 - q1
    return colname


def calc_msd(df: pd.DataFrame, period: int = 60) -> str:
    """
    移動平均差
    :param df:
    :param period:
    :return:
    """
    colname = f"MSD{period:03d}"
    df[colname] = df["Price"].rolling(period, min_periods=1).std()
    return colname


def percentile(list_data: list, p: float):
    list_data = sorted(list_data)
    k = (len(list_data) - 1) * p
    f = int(k)
    c = f + 1
    if c < len(list_data):
        return list_data[f] + (list_data[c] - list_data[f]) * (k - f)
    else:
        return list_data[f]


class EMA:
    """
    リアルタイム用の指数平滑移動平均 (Exponential Moving Average, EMA)
    """

    def __init__(self, period: int):
        self.period = period
        self.alpha = 2 / (period + 1)
        self.deque = deque(maxlen=period)
        self.ema = None  # 最新の EMA 値

    def update(self, value: float) -> float:
        # データを deque に追加
        self.deque.append(value)

        # 初期化: 最初は単純平均を使うことも可能
        if self.ema is None:
            if len(self.deque) < self.period:
                # データが period 未満なら平均で初期化
                self.ema = sum(self.deque) / len(self.deque)
            else:
                # period 個揃ったらその平均を初期 EMA に
                self.ema = sum(self.deque) / self.period
        else:
            # 通常更新: 再帰式
            self.ema = self.alpha * value + (1 - self.alpha) * self.ema

        return self.ema
