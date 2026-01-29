import datetime

import numpy as np

from funcs.technical import MovingAverage, RegressionSlope, EMA, RollingRange, MovingRange, RegressionSlopePeriod
from structs.app_enum import PositionType


class FeatureProvider:
    DEFAULTS = {
        "PERIOD_WARMUP": 60,  # 寄り付き後のウォームアップ期間
        "PERIOD_MA_1": 30,  # 短周期移動平均線の周期
        "PERIOD_MA_2": 300,  # 長周期移動平均線の周期
        "LOSSCUT_1": -25.0,  # 単純ロスカットをするためのしきい値
        "N_MINUS_MAX": 90,  # 含み損益が連続マイナスを許容する最大回数
        "DD_PROFIT": 5.0,  # 「含み益最大値」がこれを超えればドローダウン対象
        "DD_RATIO": 0.5,  # ドローダウン比率がこのしきい値を超えれば利確
    }
    INIT_VALUES = {
        # ティックデータ
        "ts": None,
        "price": None,
        "volume": None,

        # 移動平均関連
        "div_ma_prev": None,
        "cross": 0,

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
        "takeprofit": None,  # 利確フラグ
        "n_minus": None,  # 含み益が連続マイナスになったカウンタ
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
        "takeprofit": False,
        "n_minus": 0,
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

        # INIT_VALUES を一括適用
        for key, value in self.INIT_VALUES.items():
            setattr(self, key, value)

        # 変数の初期化
        self.clear()

    def clear(self):
        # オブジェクト系は個別に clear()
        self.obj_ma1.clear()
        self.obj_ma2.clear()

        # 取引明細とポジションは特殊処理
        self.dict_transaction = self.transaction_init()
        self.position = PositionType.NONE

        # 値のリセットは一括処理
        for key, value in self.CLEAR_VALUES.items():
            setattr(self, key, value)

    def doesTakeProfit(self) -> float:
        return 1 if self.takeprofit else 0

    @staticmethod
    def get_datetime(t: float) -> str:
        """
        タイムスタンプから年月日時刻へ変換（小数点切り捨て）
        :param t:
        :return:
        """
        return str(datetime.datetime.fromtimestamp(int(t)))

    def getCounterMinus(self) -> float:
        return float(self.n_minus)

    def getCrossSignal1(self) -> float:
        """
        クロスシグナル（-1, 0, 1）
        :return:
        """
        return float(self.cross)

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

    def getLosscut2(self) -> float:
        return 1 if self.n_minus > self.N_MINUS_MAX else 0

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
        損益（含み損益）の計算
        :return:
        """
        # --- 損益計算 ---
        if self.position == PositionType.LONG:
            # 返済: 買建 (LONG) → 売埋
            profit = self.price - self.price_entry
        elif self.position == PositionType.SHORT:
            # 返済: 売建 (SHORT) → 買埋
            profit = self.price_entry - self.price
        else:
            # 建玉なし
            profit = 0.0
            return profit

        # --- 最大含み益の更新 ---
        if self.profit_max < profit:
            self.profit_max = profit

        # --- ドローダウン関連の更新 ---
        if 0 < self.profit_max:
            self.drawdown = self.profit_max - profit
            self.dd_ratio = self.drawdown / self.profit_max
        else:
            self.drawdown = 0
            self.dd_ratio = 0

        # --- 含み損が続く回数カウント ---
        self.n_minus = self.n_minus + 1 if profit < 0 else 0

        return profit

    def getProfitMax(self) -> float:
        """
        最大含み益
        :return:
        """
        return self.profit_max

    def getSetting(self) -> dict:
        return self.dict_setting

    def getTimestamp(self) -> float:
        return self.ts

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

        # 取引明細更新（建玉返済）
        self.transaction_close(profit)

        # エントリ価格をリセット
        self.price_entry = 0.0

        # 含み益関連のインスタンス変数をリセット
        self.profit_max = 0.0
        self.drawdown = 0.0
        self.dd_ratio = 0.0
        self.n_minus = 0

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
        # --- クロス判定 ---
        self.cross_pre = self.cross
        self.cross = self._detect_cross(self.div_ma_prev, div_ma)
        self.div_ma_prev = div_ma

        # --- ロスカット判定 ---
        self.losscut_1 = self.getProfit() <= self.LOSSCUT_1

        # --- 利確判定 ---
        self.takeprofit = (
                self.dd_ratio > self.DD_RATIO
                and self.profit_max > self.DD_PROFIT
        )

    def _detect_cross(self, prev: float | None, curr: float) -> int:
        """移動平均の乖離の符号変化からクロスを検出"""
        if prev is None:
            return 0
        if prev < 0 < curr:
            return +1
        if curr < 0 < prev:
            return -1
        return 0

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
