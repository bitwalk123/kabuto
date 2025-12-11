import datetime
import math
from collections import deque

import numpy as np

from structs.app_enum import SignalSign, PositionType


class FeatureProvider:
    def __init__(self, dict_param: dict):
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # 定数
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        print("パラメータ")
        for key in dict_param.keys():
            print(f"{key} : {dict_param[key]}")
        # 移動平均差用（定数）
        key = "PERIOD_MA_1"
        self.PERIOD_MA_1 = dict_param.get(key, 75)
        key = "PERIOD_MA_2"
        self.PERIOD_MA_2 = dict_param.get(key, 540)
        # 移動範囲用（定数）
        key = "PERIOD_MR"
        self.PERIOD_MR = dict_param.get(key, 30)
        key = "THRESHOLD_MR"
        #self.THRESHOLD_MR = dict_param.get(key, 7)
        self.THRESHOLD_MR = dict_param.get(key, 3.5) # 8306 の呼び値は 0.5
        """
        print(
            "パラメータ",
            self.PERIOD_MA_1,
            self.PERIOD_MA_2,
            self.PERIOD_MR,
            self.THRESHOLD_MR,
        )
        """
        # ロスカット
        self.LOSSCUT_1 = 0
        self.LOSSCUT_2 = -5
        # ---------------------------------------------------------------------
        # 株価キューの最大値
        self.N_DEQUE_PRICE = self.PERIOD_MA_2
        # 最大取引回数（買建、売建）
        self.N_TRADE_MAX = 100
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # 変数
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # リアルタイムで取得する変数
        self.ts = None
        self.price = None
        self.volume = None
        self.vwap = None
        # 移動平均差用変数
        self.ma_1 = None
        self.ma_2 = None
        self.mad = None
        self.mad_sign_current = None
        self.mad_sign_signal = None
        # 移動範囲用変数
        self.mr = None
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
        self.dict_transaction = None  # 取引履歴
        self.position = None  # ポジション
        self.pnl_total = None  # 損益合計
        self.price_tick = None  # 呼び値
        self.price_entry = None  # エントリ価格
        self.profit_max = None  # 最大含み益
        self.unit = None  # 売買単位
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
        self.ma_1 = 0
        self.ma_2 = 0
        self.mad = 0
        self.mad_sign_current = SignalSign.ZERO
        self.mad_sign_signal = SignalSign.ZERO
        # 移動範囲用変数
        self.mr = 0
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
        # print("self.N_DEQUE_PRICE", self.N_DEQUE_PRICE, type(self.N_DEQUE_PRICE))
        self.deque_price = deque(maxlen=self.N_DEQUE_PRICE)  # for MA
        # ---------------------------------------------------------------------
        # 取引関連の変数
        # ---------------------------------------------------------------------
        self.dict_transaction = self.transaction_init()  # 取引明細
        self.position = PositionType.NONE  # ポジション（建玉）
        self.pnl_total = 0.0  # 総損益
        self.price_tick: float = 1.0  # 呼び値
        self.price_entry = 0.0  # 取得価格
        self.profit_max = 0.0  # 含み損益の最大値
        self.unit: float = 1  # 売買単位

    def _calc_mad(self) -> tuple[float, SignalSign]:
        """
        移動平均差 (Moving Average Difference = MAD) の算出
        :return:
        """
        mad_new = self.ma_1 - self.ma_2
        if 0 < mad_new:
            signal_sign_new = SignalSign.POSITIVE
        elif mad_new < 0:
            signal_sign_new = SignalSign.NEGATIVE
        else:
            signal_sign_new = SignalSign.ZERO

        return mad_new, signal_sign_new

    def _calc_mr(self) -> float:
        """
        移動範囲, Moving Range の算出
        :return:
        """
        n_deque = len(self.deque_price)
        if n_deque < self.PERIOD_MR:
            if n_deque > 0:
                prices_all = list(self.deque_price)
                price_min = min(prices_all)
                price_max = max(prices_all)
                return price_max - price_min
            else:
                return 0.0
        else:
            prices_recent = list(self.deque_price)[-self.PERIOD_MR:]
            price_min = min(prices_recent)
            price_max = max(prices_recent)
            return price_max - price_min

    def _calc_vwap(self) -> float:
        """
        売買高加重平均価格, Volume Weighted Average Price の算出
        :return:
        """
        if self.volume_prev is None:
            diff_volume = 0.0
        else:
            diff_volume = self.volume - self.volume_prev

        self.cum_pv += self.price * diff_volume
        self.cum_vol += diff_volume
        self.volume_prev = self.volume

        return self.cum_pv / self.cum_vol if self.cum_vol > 0 else self.price

    def doesLossCut(self) -> int:
        profit = self.get_profit()
        if -self.LOSSCUT_2 < self.profit_max and profit < self.LOSSCUT_1:
            return 1
        elif profit < self.LOSSCUT_2:
            return 1
        else:
            return 0

    '''
    def doesTakeProfit(self) -> bool:
        """
        利確
        :return:
        """
        profit = self.get_profit()
        if 50 < self.profit_max:
            if profit < self.profit_max * 0.9:
                return True
            else:
                return False
        elif 20 < self.profit_max:
            if profit < self.profit_max * 0.8:
                return True
            else:
                return False
        elif 10 < self.profit_max:
            if profit < self.profit_max * 0.3:
                return True
            else:
                return False
        else:
            return False
    '''

    def beta_ratio(
            self,
            x: float,
            A: float = 33 / 35,  # ≈ 0.942857
            B: float = 1.836735,
            C: float = -100 / 14  # ≈ -7.142857
    ):
        return A - B / (C + x)

    def take_profit_threshold(
            self,
            A: float = 33 / 35,
            B: float = 1.836735,
            C: float = -100 / 14
    ):
        return self.beta_ratio(self.profit_max, A, B, C) * self.profit_max

    def doesTakeProfit(
            self,
            Tmin: float = 10.0,
            eps_ratio: float = 0.0,
            A: float = 33 / 35,
            B: float = 1.836735,
            C: float = -100 / 14
    ) -> bool:
        profit = self.get_profit()
        if self.profit_max <= Tmin:
            return False
        threshold = self.take_profit_threshold(A, B, C)
        margin = eps_ratio * self.profit_max
        return profit < (threshold - margin)

    @staticmethod
    def get_datetime(t: float) -> str:
        """
        タイムスタンプから年月日時刻へ変換（小数点切り捨て）
        :param t:
        :return:
        """
        return str(datetime.datetime.fromtimestamp(int(t)))

    def get_profit(self) -> float:
        """
        損益計算（含み損益）
        :return:
        """
        if self.position == PositionType.LONG:
            # 返済: 買建 (LONG) → 売埋
            profit = self.price - self.price_entry
        elif self.position == PositionType.SHORT:
            # 返済: 売建 (SHORT) → 買埋
            profit = self.price_entry - self.price
        else:
            profit = 0.0

        # 最大含み益を保持
        if self.profit_max < profit:
            self.profit_max = profit

        return profit

    def get_profit_max(self) -> float:
        """
        最大含み益
        :return:
        """
        return self.profit_max

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

    def getMA1(self) -> float:
        """
        移動平均 1 の取得
        :return:
        """
        return self.ma_1

    def getMA2(self) -> float:
        """
        移動平均 2 の取得
        :return:
        """
        return self.ma_2

    def getMAD(self) -> float:
        """
        移動平均差 (Moving Average Devisation = MAD)
        """
        return self.mad

    def getMADSignal(self) -> SignalSign:
        """
        移動平均差 (Moving Average Deviation = MAD)
        上抜け、下抜けシグナル
        """
        return self.mad_sign_signal

    def getMR(self) -> float:
        """
        移動範囲 (Moving Range = MR)
        """
        return self.mr

    def getPriceRatio(self) -> float:
        """
        （始値で割った）株価比
        """
        return self.price / self.price_open if self.price_open > 0 else 0.0

    def getVWAP(self) -> float:
        """
        売買高加重平均価格, Volume Weighted Average Price を取得
        :return:
        """
        return self.vwap

    def getVWAPdr(self) -> float:
        """
        売買高加重平均価格, Volume Weighted Average Price の乖離度を取得
        :return:
        """
        if self.vwap == 0.0:
            return 0.0
        else:
            return (self.price - self.vwap) / self.vwap

    def isLowVolatility(self) -> bool:
        if self.getMR() < self.THRESHOLD_MR:
            return True
        else:
            return False

    def position_close(self) -> float:
        reward = 0

        # HOLD カウンター（建玉あり）のリセット
        self.n_hold_position = 0
        # 取引回数のインクリメント
        self.n_trade += 1

        # 確定損益
        profit = self.get_profit()
        # 確定損益追加
        self.pnl_total += profit
        # 報酬に追加
        reward += profit

        # エントリ価格をリセット
        self.price_entry = 0.0
        # 含み損益の最大値
        self.profit_max = 0.0

        # 取引明細更新（建玉返済）
        self.transaction_close(profit)

        # ポジションの更新
        self.position = PositionType.NONE

        return reward

    def position_open(self, position: PositionType) -> float:
        """
        新規ポジション
        :return:
        """
        reward = 0.0

        # HOLD カウンター（建玉なし）のリセット
        self.n_hold = 0
        # 取引回数のインクリメント
        self.n_trade += 1

        # エントリ価格
        self.price_entry = self.price
        # ポジションを更新
        self.position = position
        # 取引明細更新（新規建玉）
        self.transaction_open()

        return reward

    def resetHoldCounter(self):
        self.n_hold = 0.0  # 建玉なしの HOLD カウンタ

    def resetHoldPosCounter(self):
        self.n_hold_position = 0.0  # 建玉ありの HOLD カウンタ

    def resetTradeCounter(self):
        self.n_trade = 0.0  # 取引カウンタ

    def setCode(self, code: str):
        """
        銘柄コードの設定
        :param code:
        :return:
        """
        self.code = code

    def threshold(self) -> float:
        # 最大含み益が小さいときは粘る、大きいときは早めに利確
        # 例: ロジスティック関数で滑らかに変化
        # K = 0.05
        K = 0.0001
        C = 50
        return 1.0 - 1.0 / (1.0 + math.exp(-K * (self.profit_max - C)))

    def update(self, ts, price, volume):
        # 最新ティック情報を保持
        self.ts = ts
        if self.price_open == 0.0:
            """
            寄り付いた最初の株価が基準価格
            ※ 寄り付き後の株価が送られてくることをシステムが保証している
            """
            self.price_open = price
        # ---------------------------------------------------------------------
        # ティックデータ
        self.price = float(price)
        self.deque_price.append(float(price))  # キューへの追加
        self.volume = float(volume)
        self.vwap = self._calc_vwap()  # VWAP
        # ---------------------------------------------------------------------
        # 移動平均
        self.ma_1 = self.getMA(self.PERIOD_MA_1)
        self.ma_2 = self.getMA(self.PERIOD_MA_2)
        self.mad, mad_sign = self._calc_mad()
        if self.mad_sign_current != mad_sign:
            self.mad_sign_signal = mad_sign
        else:
            self.mad_sign_signal = SignalSign.ZERO
        self.mad_sign_current = mad_sign
        # ---------------------------------------------------------------------
        # 移動範囲
        self.mr = self._calc_mr()

    def transaction_add(self, transaction: str, profit: float = np.nan):
        """
        取引明細用データ辞書の更新
        :param transaction:
        :param profit:
        :return:
        """
        self.dict_transaction["注文日時"].append(self.get_datetime(self.ts))
        self.dict_transaction["銘柄コード"].append(self.code)
        self.dict_transaction["売買"].append(transaction)
        self.dict_transaction["約定単価"].append(self.price)
        self.dict_transaction["約定数量"].append(self.unit)
        self.dict_transaction["損益"].append(profit)

    def transaction_close(self, profit):
        """
        建玉返済時の取引明細更新
        :return:
        """
        if self.position == PositionType.LONG:
            # 返済: 買建 (LONG) → 売埋
            self.transaction_add("売埋", profit)
        elif self.position == PositionType.SHORT:
            # 返済: 売建 (SHORT) → 買埋
            self.transaction_add("買埋", profit)
        else:
            raise TypeError(f"Unknown PositionType: {self.position}")

    @staticmethod
    def transaction_init() -> dict:
        """
        取引明細用データ辞書の初期化
        :return:
        """
        return {
            "注文日時": [],
            "銘柄コード": [],
            "売買": [],
            "約定単価": [],
            "約定数量": [],
            "損益": [],
        }

    def transaction_open(self):
        """
        新規建玉時の取引明細更新
        :return:
        """
        if self.position == PositionType.LONG:
            self.transaction_add("買建")
        elif self.position == PositionType.SHORT:
            self.transaction_add("売建")
        else:
            raise TypeError(f"Unknown PositionType: {self.position}")
