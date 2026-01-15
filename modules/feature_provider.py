import datetime

import numpy as np

from funcs.technical import MovingAverage, RegressionSlope, EMA, RollingRange
from structs.app_enum import PositionType


class FeatureProvider:
    DEFAULTS = {
        "PERIOD_WARMUP": 60,
        "PERIOD_MA_1": 60,
        "PERIOD_MA_2": 600,
        "PERIOD_SLOPE": 5,
        "THRESHOLD_SLOPE": 1.0,
        "PERIOD_RR": 30,
        "TURBULENCE": 20,
        "LOSSCUT_1": -25.0,
        "THRESHOLD_PM_MIN": 10.0,
        "THRESHOLD_DDR_MIN": 0.25,
    }
    INIT_VALUES = {
        # ティックデータ
        "ts": None,
        "price": None,
        "volume": None,

        # 移動平均関連
        "div_ma_prev": None,
        "cross": 0,
        "cross_pre": 0,
        "cross_strong": False,

        # ボラティリティ関連
        "rr": 0,
        "rr_pre": 0,
        "fluc": 0,

        # ロスカット
        "losscut_1": False,

        # 始値
        "price_open": None,

        # カウンタ
        "n_trade": 0,
        "n_hold": None,
        "n_hold_position": None,

        # 取引関連
        "code": None,
        "dict_transaction": None,
        "position": None,
        "pnl_total": None,
        "price_tick": None,
        "price_entry": None,
        "profit_max": None,
        "drawdown": None,
        "dd_ratio": None,
        "unit": None,
    }

    CLEAR_VALUES = {
        # リアルタイムで取得する変数
        "ts": 0,
        "price": 0,
        "volume": 0,

        # 移動平均差用
        "div_ma_prev": None,
        "cross": 0,
        "cross_pre": 0,
        "cross_strong": False,

        # ボラティリティ関連
        "rr": 0,
        "rr_pre": 0,
        "fluc": 0,

        # ロスカット
        "losscut_1": False,

        # 始値
        "price_open": 0.0,

        # カウンタ関連
        "n_trade": 0,
        "n_hold": 0.0,
        "n_hold_position": 0.0,

        # 取引関連
        "pnl_total": 0.0,
        "price_tick": 1.0,
        "price_entry": 0.0,
        "profit_max": 0.0,
        "drawdown": 0.0,
        "dd_ratio": 0.0,
        "unit": 1,
    }

    def __init__(self, dict_setting: dict):
        # DEFAULTS をベースに dict_setting を上書きして「完全版」を作る
        self.dict_setting = {**self.DEFAULTS, **dict_setting}

        print("パラメータ")
        # 属性へ反映
        for key, value in self.dict_setting.items():
            setattr(self, key, value)
            print(f"{key} : {value}")
        # 最大取引回数
        self.N_TRADE_MAX = 100

        # インスタンス生成
        self.obj_ma1 = MovingAverage(window_size=self.PERIOD_MA_1)
        self.obj_ma2 = MovingAverage(window_size=self.PERIOD_MA_2)
        self.obj_slope1 = RegressionSlope(window_size=self.PERIOD_SLOPE)
        self.obj_slope2 = RegressionSlope(window_size=self.PERIOD_SLOPE)
        self.obj_rr = RollingRange(window_size=self.PERIOD_RR)

        # INIT_VALUES を一括適用
        for key, value in self.INIT_VALUES.items():
            setattr(self, key, value)

        # 変数の初期化
        self.clear()

    def clear(self):
        # オブジェクト系は個別に clear()
        self.obj_ma1.clear()
        self.obj_ma2.clear()
        self.obj_slope1.clear()
        self.obj_slope2.clear()
        self.obj_rr.clear()

        # 取引明細とポジションは特殊処理
        self.dict_transaction = self.transaction_init()
        self.position = PositionType.NONE

        # 値のリセットは一括処理
        for key, value in self.CLEAR_VALUES.items():
            setattr(self, key, value)

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

    def getPeriodWarmup(self):
        return self.PERIOD_WARMUP

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
        if 0 < profit:
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

    def getSetting(self) -> dict:
        return self.dict_setting

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
