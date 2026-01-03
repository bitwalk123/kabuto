from collections import deque

import pandas as pd
from scipy.stats import iqr


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


def calc_mr(df: pd.DataFrame, period: int = 60):
    """
    移動範囲 Moving Range
    :param df:
    :param period:
    :return:
    """
    colname = f"MR{period:03d}"
    ser_min = df["Price"].rolling(period, min_periods=1).min()
    ser_max = df["Price"].rolling(period, min_periods=1).max()
    df[colname] = ser_max - ser_min
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


class MovingAverage:
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.queue_data = deque(maxlen=window_size)
        self.value_ma: float = 0

    def clear(self):
        self.queue_data.clear()

    def getValue(self) -> float:
        # 移動平均を返す
        return self.value_ma

    def update(self, value: float) -> float:
        # 新しい値を追加
        self.queue_data.append(value)
        self.value_ma = sum(self.queue_data) / len(self.queue_data)
        # 移動平均を返す
        return self.value_ma


class SimpleSlope:
    """
    SimpleSlope:
    値の差分（diff）を rolling mean で平滑化した slope を逐次計算する軽量クラス。
    """

    def __init__(self, window_size: int):
        self.slope = None
        self.value_prev = None
        self.queue_data = deque(maxlen=window_size)

    def clear(self):
        self.queue_data.clear()

    def getSlope(self) -> float:
        # 移動平均を返す
        return self.slope

    def update(self, value) -> float:
        # 初回
        if self.value_prev is None:
            diff = 0.0
        else:
            diff = value - self.value_prev

        self.value_prev = value
        self.queue_data.append(diff)

        # slope = diff の rolling mean
        self.slope = sum(self.queue_data) / len(self.queue_data)
        return self.slope


class MovingRange:
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.data = deque()
        self.max_q = deque()  # 単調減少キュー（先頭が最大）
        self.min_q = deque()  # 単調増加キュー（先頭が最小）

    def clear(self):
        self.data.clear()
        self.max_q.clear()
        self.min_q.clear()

    def update(self, value: float) -> float:
        # 新しい値を追加
        self.data.append(value)

        # --- 最大値キュー更新 ---
        while self.max_q and self.max_q[-1] < value:
            self.max_q.pop()
        self.max_q.append(value)

        # --- 最小値キュー更新 ---
        while self.min_q and self.min_q[-1] > value:
            self.min_q.pop()
        self.min_q.append(value)

        # 古い値を削除（window_size を超えたら）
        if len(self.data) > self.window_size:
            old = self.data.popleft()
            if old == self.max_q[0]:
                self.max_q.popleft()
            if old == self.min_q[0]:
                self.min_q.popleft()

        # 移動範囲を返す
        return self.max_q[0] - self.min_q[0]


class PriceChangeBinary:
    def __init__(self, window_size: int):
        """
        window_size: 直近何個の価格変化イベントを保持するか
        """
        self.window_size = window_size
        self.events = deque(maxlen=window_size)
        self.last_price = None

    def update(self, price):
        if self.last_price is None:
            self.last_price = price
            return 0

        # diff = 0 → 0、diff != 0 → 1
        event = 1 if price != self.last_price else 0
        self.events.append(event)

        self.last_price = price

        # ボラティリティ = 直近 window_size 個の変化イベント数
        return sum(self.events)


class PriceChangeMedian:
    def __init__(self, window_size: int):
        """
        window_size: 直近何個の価格変化イベントを保持するか
        """
        self.window_size = window_size
        self.events = deque(maxlen=window_size)
        self.last_price = None

    def update(self, price):
        if self.last_price is None:
            self.last_price = price
            return 0

        event = abs(price - self.last_price)
        self.events.append(event)

        self.last_price = price

        return iqr(self.events)


class CappedTrendAccumulator:
    def __init__(self, window_size: int, cap: float = 2.0):
        """
        window_size: 直近何個の価格変化イベントを保持するか
        cap: 1イベントあたりの最大変化量（スパイク除去用）
        """
        self.window_size = window_size
        self.cap = cap
        self.events = deque()
        self.sum_events = 0.0
        self.last_price = None

    def update(self, price) -> float:
        if self.last_price is None:
            self.last_price = price
            return 0.0

        diff = price - self.last_price
        clipped_diff = max(min(diff, self.cap), -self.cap)

        self.events.append(clipped_diff)
        self.sum_events += clipped_diff
        self.last_price = price

        if len(self.events) > self.window_size:
            removed = self.events.popleft()
            self.sum_events -= removed

        return abs(self.sum_events)
