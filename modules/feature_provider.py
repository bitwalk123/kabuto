from collections import deque
from statistics import stdev

from structs.app_enum import SignalSign


class FeatureProvider:
    def __init__(self):
        self.ts = 0
        self.price = 0
        self.volume = 0
        self.vwap = 0
        self.mad = 0
        self.mad_sign_current = SignalSign.ZERO
        self.mad_sign_signal = SignalSign.ZERO
        self.msd = 0

        # 特徴量算出のために保持する変数
        self.price_open = 0.0  # ザラバの始値
        self.cum_pv = 0.0  # VWAP 用 Price × Volume 累積
        self.cum_vol = 0.0  # VWAP 用 Volume 累積
        self.volume_prev = None  # VWAP 用 前の Volume

        # カウンタ関連
        self.n_trade_max = 100.0  # 最大取引回数（買建、売建）
        self.n_trade = 0.0  # 取引カウンタ
        self.n_hold = 0.0  # 建玉なしの HOLD カウンタ
        self.n_hold_position = 0.0  # 建玉ありの HOLD カウンタ

        # キューを定義
        self.n_deque_price = 600
        self.deque_price = deque(maxlen=self.n_deque_price)  # for MA

    def _calc_mad(self) -> tuple[float, SignalSign]:
        """
        移動平均差 (Moving Average Difference = MAD)
        :return:
        """
        ma_060 = self.getMA(60)
        ma_300 = self.getMA(600)
        mad_new = ma_060 - ma_300
        if 0 < mad_new:
            signal_sign_new = SignalSign.POSITIVE
        elif mad_new < 0:
            signal_sign_new = SignalSign.NEGATIVE
        else:
            signal_sign_new = SignalSign.ZERO

        return mad_new, signal_sign_new

    def _calc_msd(self) -> float:
        """
        移動標準偏差 (Moving Standard Deviation = MSD)
        """
        n_deque = len(self.deque_price)
        if n_deque < self.period_msd:
            return stdev(list(self.deque_price)) if n_deque > 1 else 0.0
        else:
            recent_prices = list(self.deque_price)[-self.period_msd:]
            return stdev(recent_prices)

    def _calc_vwap(self) -> float:
        if self.volume_prev is None:
            diff_volume = 0.0
        else:
            diff_volume = self.volume - self.volume_prev

        self.cum_pv += self.price * diff_volume
        self.cum_vol += diff_volume
        self.volume_prev = self.volume

        return self.cum_pv / self.cum_vol if self.cum_vol > 0 else self.price

    def clear(self):
        self.ts = 0
        self.price = 0
        self.volume = 0
        self.vwap = 0
        self.mad = 0
        self.mad_sign_current = SignalSign.ZERO
        self.mad_sign_signal = SignalSign.ZERO
        self.msd = 0

        # 特徴量算出のために保持する変数
        self.price_open = 0.0  # ザラバの始値
        self.cum_pv = 0.0  # VWAP 用 Price × Volume 累積
        self.cum_vol = 0.0  # VWAP 用 Volume 累積
        self.volume_prev = None  # VWAP 用 前の Volume
        self.period_msd = 60

        # カウンタ関連
        # 取引カウンタ
        self.resetTradeCounter()
        # 建玉なしの HOLD カウンタ
        self.resetHoldCounter()
        # 建玉ありの HOLD カウンタ
        self.resetHoldPosCounter()

        # キュー
        self.deque_price.clear()  # 移動平均など

    def getMA(self, period: int) -> float:
        """
        移動平均 (Moving Average = MA)
        """
        n_deque = len(self.deque_price)
        if n_deque < period:
            return sum(self.deque_price) / n_deque if n_deque > 0 else 0.0
        else:
            recent_prices = list(self.deque_price)[-period:]
            return sum(recent_prices) / period

    def getMAD(self) -> float:
        """
        移動平均差 (Moving Average Devisation = MAD)
        """
        return self.mad

    def getMADSignal(self) -> float:
        """
        移動平均差 (Moving Average Devisation = MAD)
        """
        return float(self.mad_sign_signal.value)

    def getMSD(self) -> float:
        """
        移動標準偏差 (Moving Standard Deviation = MSD)
        """
        return self.msd

    def getPriceRatio(self) -> float:
        """
        （始値で割った）株価比
        """
        return self.price / self.price_open if self.price_open > 0 else 0.0

    def getVWAPdr(self) -> float:
        if self.vwap == 0.0:
            return 0.0
        else:
            return (self.price - self.vwap) / self.vwap

    def resetHoldCounter(self):
        self.n_hold = 0.0  # 建玉なしの HOLD カウンタ

    def resetHoldPosCounter(self):
        self.n_hold_position = 0.0  # 建玉ありの HOLD カウンタ

    def resetTradeCounter(self):
        self.n_trade = 0.0  # 取引カウンタ

    def update(self, ts, price, volume):
        # 最新ティック情報を保持
        self.ts = ts
        if self.price_open == 0.0:
            """
            寄り付いた最初の株価が基準価格
            ※ 寄り付き後の株価が送られてくることをシステムが保証している
            """
            self.price_open = price

        self.price = float(price)
        self.deque_price.append(float(price))  # キューへの追加
        self.volume = float(volume)
        self.vwap = self._calc_vwap()  # VWAP
        self.mad, mad_sign = self._calc_mad() # 移動平均差
        if self.mad_sign_current != mad_sign:
            self.mad_sign_signal = mad_sign
        else:
            self.mad_sign_signal = SignalSign.ZERO
        self.mad_sign_current = mad_sign
        self.msd = self._calc_msd()  # 移動標準偏差
