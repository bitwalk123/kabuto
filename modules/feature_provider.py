import datetime

import numpy as np

from funcs.technical import MovingAverage, RegressionSlope, EMA, RollingRange
from structs.app_enum import PositionType


class FeatureProvider:
    def __init__(self, dict_setting: dict):
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # 定数
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        print("パラメータ")
        for key in dict_setting.keys():
            print(f"{key} : {dict_setting[key]}")
        # ---------------------------------------------------------------------
        # 1. ウォームアップ期間
        key = "PERIOD_WARMUP"
        self.PERIOD_WARMUP: int = dict_setting.get(key, 60)
        # 2. MA_1 移動平均期間
        key = "PERIOD_MA_1"
        self.PERIOD_MA_1: int = dict_setting.get(key, 60)
        # 3. MA_2 移動平均期間
        key = "PERIOD_MA_2"
        self.PERIOD_MA_2: int = dict_setting.get(key, 600)
        # 4. MA_1 の傾き（軽い平滑化期間）
        key = "PERIOD_SLOPE"
        self.PERIOD_SLOPE: int = dict_setting.get(key, 5)
        # 5. クロス時の MA_1 の傾きの閾値
        key = "THRESHOLD_SLOPE"
        self.THRESHOLD_SLOPE: float = dict_setting.get(key, 1.0)  # doe-10
        # 6. RollingRange
        key = "PERIOD_RR"
        self.PERIOD_RR: int = dict_setting.get(key, 30)
        # 7. Turbulence（乱高下）
        key = "TURBULENCE"
        self.TURBULENCE: int = dict_setting.get(key, 20)
        # 8. 単純ロスカットの閾値 1
        key = "LOSSCUT_1"
        self.LOSSCUT_1: float = dict_setting.get(key, -25.0)
        # ---------------------------------------------------------------------
        # 最大取引回数（買建、売建）
        self.N_TRADE_MAX = 100
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # 変数とインスタンス
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # ティックデータ
        self.ts = None
        self.price = None
        self.volume = None
        # ---------------------------------------------------------------------
        # 移動平均
        self.obj_ma1 = MovingAverage(window_size=self.PERIOD_MA_1)
        self.obj_ma2 = MovingAverage(window_size=self.PERIOD_MA_2)
        self.obj_slope1 = RegressionSlope(window_size=self.PERIOD_SLOPE)
        self.obj_slope2 = RegressionSlope(window_size=self.PERIOD_SLOPE)
        self.div_ma_prev = None
        self.cross = 0
        self.cross_pre = 0
        self.cross_strong = False
        # ---------------------------------------------------------------------
        # ボラティリティ関連
        self.obj_rr = RollingRange(window_size=self.PERIOD_RR)
        self.rr = 0
        self.rr_pre = 0
        self.fluc = 0
        # ---------------------------------------------------------------------
        # ロスカット
        self.losscut_1 = False
        # ---------------------------------------------------------------------
        # 始値
        self.price_open = None
        # ---------------------------------------------------------------------
        # カウンタ関連
        self.n_trade = 0
        self.n_hold = None
        self.n_hold_position = None
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
        self.drawdown = None  # ドローダウン
        self.dd_ratio = None  # ドローダウン比率
        self.unit = None  # 売買単位
        # ---------------------------------------------------------------------
        # 変数の初期化
        self.clear()

    def clear(self):
        # リアルタイムで取得する変数
        self.ts = 0
        self.price = 0
        self.volume = 0
        # 移動平均差用
        self.obj_ma1.clear()
        self.obj_ma2.clear()
        self.obj_slope1.clear()
        self.obj_slope2.clear()
        self.div_ma_prev = None
        self.cross = 0
        self.cross_pre = 0
        self.cross_strong = False
        # ボラティリティ関連
        self.obj_rr.clear()
        self.rr = 0
        self.rr_pre = 0
        self.fluc = 0
        # ロスカット
        self.losscut_1 = False
        # 始値
        self.price_open: float = 0  # ザラバの始値
        # カウンタ関連
        self.n_trade = 0  # 取引カウンタ
        self.n_hold = 0.0  # 建玉なしの HOLD カウンタ
        self.n_hold_position = 0.0  # 建玉ありの HOLD カウンタ
        # ---------------------------------------------------------------------
        # 取引関連の変数
        # ---------------------------------------------------------------------
        self.dict_transaction = self.transaction_init()  # 取引明細
        self.position = PositionType.NONE  # ポジション（建玉）
        self.pnl_total = 0.0  # 総損益
        self.price_tick: float = 1.0  # 呼び値
        self.price_entry = 0.0  # 取得価格
        self.profit_max = 0.0  # 含み損益の最大値
        self.drawdown = 0.0  # ドローダウン
        self.dd_ratio = 0.0  # ドローダウン比率
        self.unit: float = 1  # 売買単位

    @staticmethod
    def get_datetime(t: float) -> str:
        """
        タイムスタンプから年月日時刻へ変換（小数点切り捨て）
        :param t:
        :return:
        """
        return str(datetime.datetime.fromtimestamp(int(t)))

    def getCrossSignal1(self) -> float:
        """
        クロスシグナル（-1, 0, 1）
        :return:
        """
        return float(self.cross)

    def getCrossSignal2(self) -> float:
        """
        クロスシグナル（-1, 0, 1）
        :return:
        """
        return float(self.cross_pre)

    def getCrossSignalStrength(self) -> float:
        """
        強いクロスシグナルか？（0: なし/弱い、1: 強い）
        :return:
        """
        return 1 if self.cross_strong else 0

    def getDDRatio(self) -> float:
        """
        ドローダウン
        :return:
        """
        return self.dd_ratio

    def getDrawDown(self) -> float:
        # ドローダウン比率
        return self.drawdown

    def getLosscut1(self) -> float:
        return 1 if self.losscut_1 else 0

    def getMA1(self) -> float:
        """
        移動平均 1 の取得
        :return:
        """
        return self.obj_ma1.getValue()

    def getMA2(self) -> float:
        """
        移動平均 2 の取得
        :return:
        """
        return self.obj_ma2.getValue()

    def getPositionValue(self) -> float:
        return float(self.position.value)

    def getPrice(self) -> float:
        return self.price

    def getProfit(self) -> float:
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

        # ドローダウン、比率算出
        if 0 < profit and 0 < self.profit_max:
            self.drawdown = self.profit_max - profit
            self.dd_ratio = self.drawdown / self.profit_max
        else:
            self.drawdown = 0.0
            self.dd_ratio = 0.0

        return profit

    def getProfitMax(self) -> float:
        """
        最大含み益
        :return:
        """
        return self.profit_max

    def getRR(self) -> float:
        return self.obj_rr.getValue()

    def getSlope1(self) -> float:
        return self.obj_slope1.getSlope()

    def getTimestamp(self) -> float:
        return self.ts

    def isFluctuation(self) -> float:
        return self.fluc

    def position_close(self) -> float:
        reward = 0

        # HOLD カウンター（建玉あり）のリセット
        self.n_hold_position = 0
        # 取引回数のインクリメント
        self.n_trade += 1

        # 確定損益
        profit = self.getProfit()
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

    def setCode(self, code: str):
        """
        銘柄コードの設定
        :param code:
        :return:
        """
        self.code = code

    def update(self, ts: float, price: float, volume: float):
        # --- 最新ティック更新 ---
        self.ts = ts
        self.price = price
        self.volume = volume

        # --- 寄り付き価格の初期化 ---
        if self.price_open == 0:
            self.price_open = price

        # --- 移動平均 ---
        ma1 = self.obj_ma1.update(price)
        ma2 = self.obj_ma2.update(price)
        div_ma = ma1 - ma2

        # --- ボラティリティ関連 ---
        self.rr_pre = self.rr
        self.rr = self.obj_rr.update(price)
        if self.rr_pre > self.TURBULENCE:
            self.fluc = 1
        elif self.rr > self.TURBULENCE:
            self.fluc = 1
        else:
            self.fluc = 0

        # --- MA1 の傾き ---
        slope1 = self.obj_slope1.update(ma1)

        # --- クロス判定 ---
        self.cross_pre = self.cross
        self.cross = self._detect_cross(self.div_ma_prev, div_ma)
        self.div_ma_prev = div_ma

        # --- クロス強度判定 ---
        self.cross_strong = self._is_cross_strong(self.cross, self.cross_pre, slope1)

        # --- ロスカット判定 ---
        self.losscut_1 = self.getProfit() <= self.LOSSCUT_1

    def _detect_cross(self, prev: float | None, curr: float) -> int:
        """移動平均の乖離の符号変化からクロスを検出"""
        if prev is None:
            return 0
        if prev < 0 < curr:
            return +1
        if curr < 0 < prev:
            return -1
        return 0

    def _is_cross_strong(self, cross: int, cross_pre: int, slope1: float) -> bool:
        """クロス強度（角度）判定"""
        if cross != 0:
            # クロス発生時
            return slope1 > self.THRESHOLD_SLOPE

        if cross_pre != 0:
            # 1秒前にクロス → 反対売買用
            return slope1 > self.THRESHOLD_SLOPE

        return False

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
