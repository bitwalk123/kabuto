from collections import deque
from typing import Optional

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
        self.ema = None

    def clear(self):
        self.ema = None

    def getValue(self) -> float:
        # 現在値を返す
        return self.ema

    def update(self, value: float) -> float:
        if self.ema is None:
            # 初期 EMA は最初の値をそのまま採用
            self.ema = value
        else:
            # 再帰式
            self.ema += self.alpha * (value - self.ema)

        return self.ema


class MovingAverageOld:
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.queue = deque(maxlen=window_size)
        self.ma: float = 0

    def clear(self):
        self.queue.clear()

    def getValue(self) -> float:
        # 現在値を返す
        return self.ma

    def update(self, value: float) -> float:
        # 新しい値を追加
        self.queue.append(value)
        self.ma = sum(self.queue) / len(self.queue)
        # 移動平均を返す
        return self.ma


class MovingAverage:
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.queue = deque()
        self.running_sum = 0.0
        self.ma = 0.0

    def clear(self):
        self.queue.clear()
        self.running_sum = 0.0
        self.ma = 0.0

    def getValue(self) -> float:
        return self.ma

    def update(self, value: float) -> float:
        # 古い値を取り除く
        if len(self.queue) == self.window_size:
            oldest = self.queue.popleft()
            self.running_sum -= oldest

        # 新しい値を追加
        self.queue.append(value)
        self.running_sum += value

        # 移動平均を計算
        self.ma = self.running_sum / len(self.queue)
        return self.ma


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

        # slope = diff の rolling mean の絶対値
        self.slope = abs(sum(self.queue_data) / len(self.queue_data))
        return self.slope


class RegressionSlope:
    """
    RegressionSlope:
    最新の点を原点とみなしたうえで、
    「原点を通る（切片0）」単回帰の傾きを逐次計算する軽量クラス。

    - x軸: 1秒間隔を仮定
    - 最新の点を (x=0, y=0) として扱う
    """

    def __init__(self, window_size: int):
        self.slope: Optional[float] = None
        self.queue_price = deque(maxlen=window_size)

    def clear(self):
        self.queue_price.clear()
        self.slope = None

    def getSlope(self) -> float:
        # 現在の傾きを返す（未計算の場合は 0.0）
        return self.slope if self.slope is not None else 0.0

    def update(self, value: float) -> float:
        """
        新しい価格を追加し、最新点を原点とする
        切片0の単回帰の傾きを計算して返す。
        """
        self.queue_price.append(value)

        n = len(self.queue_price)
        if n < 2:
            # 点が1つだけでは傾きは定義できないので0とする
            self.slope = 0.0
            return abs(self.slope)

        # 最新点を原点(0,0)とみなす
        p_last = self.queue_price[-1]

        # x = -(n-1), ..., -1, 0 となるように設定
        # y_i = p_i - p_last
        xs = [i - (n - 1) for i in range(n)]
        ys = [p - p_last for p in self.queue_price]

        sum_xy = sum(x * y for x, y in zip(xs, ys))
        sum_x2 = sum(x * x for x in xs)

        # 切片0の単回帰の傾き: m = Σ(xy) / Σ(x^2)
        self.slope = sum_xy / sum_x2 if sum_x2 != 0 else 0.0
        return abs(self.slope)


class RegressionSlopeFast:
    """
    RegressionSlope:
    最新の点を原点とみなしたうえで、
    「原点を通る（切片0）」単回帰の傾きを逐次計算するクラス。

    - x軸: 1秒間隔を仮定
    - 最新の点を (x=0, y=0) として扱う
    """

    def __init__(self, window_size: int):
        self.window_size = window_size
        self.queue_price = deque(maxlen=window_size)
        self.slope: Optional[float] = None

    def clear(self):
        self.queue_price.clear()
        self.slope = None

    def getSlope(self) -> float:
        return self.slope if self.slope is not None else 0.0

    def update(self, value: float) -> float:
        self.queue_price.append(value)

        n = len(self.queue_price)
        if n < 2:
            self.slope = 0.0
            return 0.0

        p_last = self.queue_price[-1]

        # x = -(n-1), ..., -1, 0
        # y_i = p_i - p_last
        sum_xy = 0.0
        sum_x2 = 0.0
        for i, p in enumerate(self.queue_price):
            x = i - (n - 1)
            y = p - p_last
            sum_xy += x * y
            sum_x2 += x * x

        self.slope = sum_xy / sum_x2 if sum_x2 != 0 else 0.0
        return abs(self.slope)


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


class RollingRange:
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.prices = deque(maxlen=window_size)
        self.current_range = 0.0
        self.current_high = None
        self.current_low = None

    def clear(self):
        self.prices.clear()
        self.current_range = 0.0
        self.current_high = None
        self.current_low = None

    def update(self, price: float) -> float:
        self.prices.append(price)

        # high/low を更新
        if self.current_high is None:
            # 初回
            self.current_high = price
            self.current_low = price
        else:
            # 新規値で更新
            self.current_high = max(self.current_high, price)
            self.current_low = min(self.current_low, price)

        # deque が満杯になったとき、古い値が抜けるので high/low を再計算
        if len(self.prices) == self.window_size:
            oldest = self.prices[0]
            if oldest == self.current_high or oldest == self.current_low:
                # high/low が抜けたので再計算
                self.current_high = max(self.prices)
                self.current_low = min(self.prices)

        self.current_range = self.current_high - self.current_low
        return self.current_range

    def getValue(self) -> float:
        return self.current_range


class MonotonicRange:
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.prices = deque()
        self.max_q = deque()  # 単調減少キュー（先頭が最大）
        self.min_q = deque()  # 単調増加キュー（先頭が最小）

    def clear(self):
        self.prices.clear()
        self.max_q.clear()
        self.min_q.clear()

    def update(self, price: float) -> float:
        # 新規価格を追加
        self.prices.append(price)

        # max_q 更新（単調減少）
        while self.max_q and self.max_q[-1] < price:
            self.max_q.pop()
        self.max_q.append(price)

        # min_q 更新（単調増加）
        while self.min_q and self.min_q[-1] > price:
            self.min_q.pop()
        self.min_q.append(price)

        # window サイズを超えたら古い値を削除
        if len(self.prices) > self.window_size:
            old = self.prices.popleft()
            if old == self.max_q[0]:
                self.max_q.popleft()
            if old == self.min_q[0]:
                self.min_q.popleft()

        # レンジ = 最大 - 最小
        return self.max_q[0] - self.min_q[0]

    def getValue(self) -> float:
        if not self.max_q or not self.min_q:
            return 0.0
        return self.max_q[0] - self.min_q[0]
