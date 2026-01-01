import datetime

import numpy as np

from funcs.technical import MovingAverage, SimpleSlope
from structs.app_enum import PositionType


class FeatureProvider:
    def __init__(self, dict_param: dict):
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # 定数
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        print("パラメータ")
        for key in dict_param.keys():
            print(f"{key} : {dict_param[key]}")
        # ---------------------------------------------------------------------
        # 1. MA_1 移動平均期間
        key = "PERIOD_MA_1"
        self.PERIOD_MA_1: int = dict_param.get(key, 60)
        # 2. MA_2 移動平均期間
        key = "PERIOD_MA_2"
        self.PERIOD_MA_2: int = dict_param.get(key, 600)
        # 3. MA_1 の傾き（軽い平滑化期間）
        key = "PERIOD_SLOPE"
        self.PERIOD_SLOPE: int = dict_param.get(key, 5)
        # 4. クロス時の MA_1 の傾きの閾値
        key = "THRESHOLD_SLOPE"
        self.THRESHOLD_SLOPE: float = dict_param.get(key, 0.01)
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
        # 移動平均差用
        self.obj_ma1 = MovingAverage(window_size=self.PERIOD_MA_1)
        self.obj_ma2 = MovingAverage(window_size=self.PERIOD_MA_2)
        self.obj_slope1 = SimpleSlope(window_size=self.PERIOD_SLOPE)
        self.obj_slope2 = SimpleSlope(window_size=self.PERIOD_SLOPE)
        self.div_ma_prev = None
        self.cross = 0
        self.cross_strong = False
        # 始値
        self.price_open = None
        # カウンタ関連
        self.n_trade = 0
        self.n_hold = None
        self.n_hold_position = None
        # キュー
        # self.deque_price = None
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
        # 移動平均差用
        self.obj_ma1.clear()
        self.obj_ma2.clear()
        self.obj_slope1.clear()
        self.obj_slope2.clear()
        self.div_ma_prev = None
        self.cross = 0
        self.cross_strong = False
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
        self.unit: float = 1  # 売買単位

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

    def getCrossSignal(self) -> float:
        """
        クロスシグナル（0: なし、1: あり）
        :return:
        """
        return 0 if self.cross == 0 else 1

    def getCrossSignalStrength(self) -> float:
        """
        強いクロスシグナルか？（0: なし/弱い、1: 強い）
        :return:
        """
        return 1 if self.cross_strong else 0

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

    def getTimestamp(self) -> float:
        return self.ts

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

    def setCode(self, code: str):
        """
        銘柄コードの設定
        :param code:
        :return:
        """
        self.code = code

    def update(self, ts: float, price: float, volume: float):
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # 最新ティック情報を保持
        self.ts = ts
        # ---------------------------------------------------------------------
        if self.price_open == 0:
            """
            寄り付いた最初の株価が基準価格
            ※ 寄り付き後の株価が送られてくることをシステムが保証している
            """
            self.price_open = price
        # ---------------------------------------------------------------------
        self.price = price
        self.volume = volume
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # 移動平均
        ma1 = self.obj_ma1.update(price)
        ma2 = self.obj_ma2.update(price)
        # ---------------------------------------------------------------------
        # 2つの移動平均の乖離度
        div_ma = ma1 - ma2
        # ---------------------------------------------------------------------
        # クロス判定
        if self.div_ma_prev is None:
            self.cross = 0
        else:
            if self.div_ma_prev < 0 < div_ma:
                self.cross = +1
            elif div_ma < 0 < self.div_ma_prev:
                self.cross = -1
            else:
                self.cross = 0
        self.div_ma_prev = div_ma
        # ---------------------------------------------------------------------
        # クロスの角度（強度）判定
        if self.cross == 0:
            self.cross_strong = False
        else:
            slope1 = self.obj_slope1.update(ma1)
            """
            slope2 = self.obj_slope2.update(ma2)
            # 角度の強さ（atan 不要）
            slope_diff = abs(slope1 - slope2)
            self.cross_strong = slope_diff > self.THRESHOLD_SLOPE
            """
            self.cross_strong = slope1 > self.THRESHOLD_SLOPE

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
