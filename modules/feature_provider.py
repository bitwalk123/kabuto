from collections import deque
from statistics import stdev

from funcs.technical import EMA
from structs.app_enum import SignalSign, PositionType


class FeatureProvider:
    def __init__(self):
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # 定数
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # 移動平均差用（定数）
        self.PERIOD_MAD_1 = 60
        self.PERIOD_MAD_2 = 600
        # 移動標準偏差用（定数）
        self.PERIOD_MSD = 60
        self.THRESHOLD_MSD = 3
        # 最大取引回数（買建、売建）
        self.N_TRADE_MAX = 100.0
        # 株価キューの最大値
        self.N_DEQUE_PRICE = self.PERIOD_MAD_2
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # 変数
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # リアルタイムで取得する変数
        self.ts = None
        self.price = None
        self.volume = None
        self.vwap = None
        # 移動平均差用変数
        self.mad = None
        self.mad_sign_current = None
        self.mad_sign_signal = None
        # 指数平滑移動平均差用変数
        self.emad = None
        self.emad_sign_current = None
        self.emad_sign_signal = None
        # 移動標準偏差用変数
        self.msd = None
        # 指数平滑移動平均差用インスタンス
        self.ema_1 = None
        self.ema_2 = None
        # 特徴量算出のために保持する変数
        self.price_open = None
        self.cum_pv = None
        self.cum_vol = None
        self.volume_prev = None
        # カウンタ関連
        self.n_trade = None
        self.n_hold = None
        self.n_hold_position = None
        # キュー
        self.deque_price = None
        # ---------------------------------------------------------------------
        # 取引関連の変数
        # ---------------------------------------------------------------------
        self.code = None  # 銘柄コード
        self.dict_transaction = None # 取引履歴
        self.position = None # ポジション
        self.pnl_total = None # 損益合計
        self.price_tick = None # 呼び値
        self.price_entry = None # エントリ価格
        self.profit_max = None # 最大含み益
        self.unit = None # 売買単位
        # ---------------------------------------------------------------------
        # 変数の初期化
        self.clear()

    def clear(self):
        # リアルタイムで取得する変数
        self.ts = 0
        self.price = 0
        self.volume = 0
        self.vwap = 0
        # 移動平均差用変数
        self.mad = 0
        self.mad_sign_current = SignalSign.ZERO
        self.mad_sign_signal = SignalSign.ZERO
        # 移動平均差用変数
        self.emad = 0
        self.emad_sign_current = SignalSign.ZERO
        self.emad_sign_signal = SignalSign.ZERO
        # 移動標準偏差用変数
        self.msd = 0
        # 指数平滑移動平均差用インスタンス
        self.ema_1 = EMA(self.PERIOD_MAD_1)
        self.ema_2 = EMA(self.PERIOD_MAD_2)
        # 特徴量算出のために保持する変数
        self.price_open = 0.0  # ザラバの始値
        self.cum_pv = 0.0  # VWAP 用 Price × Volume 累積
        self.cum_vol = 0.0  # VWAP 用 Volume 累積
        self.volume_prev = None  # VWAP 用 前の Volume
        # カウンタ関連
        self.n_trade = 0.0  # 取引カウンタ
        self.n_hold = 0.0  # 建玉なしの HOLD カウンタ
        self.n_hold_position = 0.0  # 建玉ありの HOLD カウンタ
        # キューを定義
        self.deque_price = deque(maxlen=self.N_DEQUE_PRICE)  # for MA
        # ---------------------------------------------------------------------
        # 取引関連の変数
        # ---------------------------------------------------------------------
        self.dict_transaction = self.init_transaction()  # 取引明細
        self.position = PositionType.NONE  # ポジション（建玉）
        self.pnl_total = 0.0  # 総損益
        self.price_tick: float = 1.0  # 呼び値
        self.price_entry = 0.0  # 取得価格
        self.profit_max = 0.0  # 含み損益の最大値
        self.unit: float = 1  # 売買単位

    def _calc_emad(self) -> tuple[float, SignalSign]:
        """
        指数平滑移動平均差 (Exponential Moving Average Difference = EMAD)
        :return:
        """
        ema_1 = self.ema_1.update(self.price)
        ema_2 = self.ema_2.update(self.price)
        emad_new = ema_1 - ema_2
        if 0 < emad_new:
            signal_sign_new = SignalSign.POSITIVE
        elif emad_new < 0:
            signal_sign_new = SignalSign.NEGATIVE
        else:
            signal_sign_new = SignalSign.ZERO

        return emad_new, signal_sign_new

    def _calc_mad(self) -> tuple[float, SignalSign]:
        """
        移動平均差 (Moving Average Difference = MAD)
        :return:
        """
        ma1 = self.getMA(self.PERIOD_MAD_1)
        ma2 = self.getMA(self.PERIOD_MAD_2)
        mad_new = ma1 - ma2
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
        if n_deque < self.PERIOD_MSD:
            return stdev(list(self.deque_price)) if n_deque > 1 else 0.0
        else:
            recent_prices = list(self.deque_price)[-self.PERIOD_MSD:]
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

    def getEMAD(self) -> float:
        """
        移動EMA差 (EMA Devisation = MAD)
        """
        return self.emad

    def getEMADSignal(self) -> float:
        """
        移動平均差 (Moving Average Devisation = MAD)
        """
        return float(self.emad_sign_signal.value)

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

    @staticmethod
    def init_transaction() -> dict:
        return {
            "注文日時": [],
            "銘柄コード": [],
            "売買": [],
            "約定単価": [],
            "約定数量": [],
            "損益": [],
        }

    def isLowVolatility(self) -> bool:
        if self.getMSD() < self.THRESHOLD_MSD:
            return True
        else:
            return False

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

        self.mad, mad_sign = self._calc_mad()  # 移動平均差
        if self.mad_sign_current != mad_sign:
            self.mad_sign_signal = mad_sign
        else:
            self.mad_sign_signal = SignalSign.ZERO
        self.mad_sign_current = mad_sign
        """
        self.emad, emad_sign = self._calc_emad()  # EMA差
        if self.emad_sign_current != emad_sign:
            self.emad_sign_signal = emad_sign
        else:
            self.emad_sign_signal = SignalSign.ZERO
        self.emad_sign_current = emad_sign
        """

        self.msd = self._calc_msd()  # 移動標準偏差
