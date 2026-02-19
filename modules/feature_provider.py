import datetime
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from modules.technical import MovingAverage, VWAP
from structs.defaults import FeatureDefaults
from structs.app_enum import PositionType


def _transaction_init() -> dict[str, list]:
    """取引明細用データ辞書の初期化（モジュールレベル関数）"""
    return {
        "注文日時": [],
        "銘柄コード": [],
        "売買": [],
        "約定単価": [],
        "約定数量": [],
        "損益": [],
    }


@dataclass
class FeatureState:
    """
    FeatureProvider が管理する状態変数をまとめた dataclass。
    パラメータの追加・削除はここだけ変更すればよい。
    """
    # ティックデータ
    ts: float = 0.0
    price: float = 0.0
    volume: float = 0.0

    # 移動平均関連
    div_ma_prev: float | None = None
    cross: float = 0.0

    # ロスカット
    losscut_1: bool = False

    # 始値
    price_open: float = 0.0

    # カウンタ
    step_current: int = 0
    n_trade: int = 0
    n_hold: int = 0
    n_hold_position: int = 0

    # 取引関連
    code: str | None = None
    dict_transaction: dict[str, list] = field(default_factory=_transaction_init)
    position: PositionType = PositionType.NONE
    pnl_total: float = 0.0
    price_tick: float = 1.0
    price_entry: float = 0.0
    profit_max: float = 0.0
    drawdown: float = 0.0
    dd_ratio: float = 0.0
    takeprofit: bool = False
    n_minus: int = 0
    unit: int = 1


class FeatureProvider:

    def __init__(self, dict_setting: dict) -> None:
        # defaults をベースに dict_setting を上書きして「完全版」を作る
        defaults = FeatureDefaults.as_dict()
        self.dict_setting: dict[str, Any] = {**defaults, **dict_setting}

        print("パラメータ")
        for key, value in self.dict_setting.items():
            setattr(self, key, value)
            print(f"{key} : {value}")

        # 最大取引回数
        self.N_TRADE_MAX: int = 100

        # インスタンス生成
        self.obj_vwap = VWAP()
        self.obj_ma1 = MovingAverage(window_size=self.PERIOD_MA_1)

        self.s: FeatureState = FeatureState()

        # 状態変数の初期化
        self.clear()

    def clear(self) -> None:
        # オブジェクト系は個別に clear()
        self.obj_vwap.clear()
        self.obj_ma1.clear()

        # FeatureState を新規生成するだけでリセット完了
        self.s = FeatureState()

    # ------------------------------------------------------------------
    # プロパティ取得系
    # ------------------------------------------------------------------

    def doesTakeProfit(self) -> float:
        return 1.0 if self.s.takeprofit else 0.0

    @staticmethod
    def get_datetime(t: float) -> str:
        """タイムスタンプから年月日時刻へ変換（小数点切り捨て）"""
        return str(datetime.datetime.fromtimestamp(int(t)))

    def getCounterMinus(self) -> float:
        return float(self.s.n_minus)

    def getCrossSignal1(self) -> float:
        """クロスシグナル（-1.0, 0.0, 1.0）"""
        return float(self.s.cross)

    def getDDRatio(self) -> float:
        """ドローダウン比率"""
        return self.s.dd_ratio

    def getDrawDown(self) -> float:
        """ドローダウン"""
        return self.s.drawdown

    def getLosscut1(self) -> float:
        return 1.0 if self.s.losscut_1 else 0.0

    def getLosscut2(self) -> float:
        return 1.0 if self.s.n_minus > self.N_MINUS_MAX else 0.0

    def getMA1(self) -> float:
        """移動平均 1 の取得"""
        return self.obj_ma1.getValue()

    def getNTrade(self) -> int:
        return self.s.n_trade

    """
    def getLower(self) -> float:
        return self.obj_miqr.getLower()

    def getUpper(self) -> float:
        return self.obj_miqr.getUpper()
    """

    def getPeriodWarmup(self) -> int:
        return self.PERIOD_WARMUP

    def getCurrentPosition(self) -> PositionType:
        return self.s.position

    def getPositionValue(self) -> float:
        return float(self.s.position.value)

    def getPrice(self) -> float:
        return self.s.price

    def getProfit(self) -> float:
        """損益（含み損益）の計算"""
        # --- 損益計算 ---
        if self.s.position == PositionType.LONG:
            # 返済: 買建 (LONG) → 売埋
            profit = self.s.price - self.s.price_entry
        elif self.s.position == PositionType.SHORT:
            # 返済: 売建 (SHORT) → 買埋
            profit = self.s.price_entry - self.s.price
        else:
            # 建玉なし
            return 0.0

        # --- 最大含み益の更新 ---
        if self.s.profit_max < profit:
            self.s.profit_max = profit

        # --- ドローダウン関連の更新 ---
        if 0 < self.s.profit_max:
            self.s.drawdown = self.s.profit_max - profit
            self.s.dd_ratio = self.s.drawdown / self.s.profit_max
        else:
            self.s.drawdown = 0.0
            self.s.dd_ratio = 0.0

        # --- 含み損が続く回数カウント ---
        self.s.n_minus = self.s.n_minus + 1 if profit < 0 else 0

        return profit

    def getProfitMax(self) -> float:
        """最大含み益"""
        return self.s.profit_max

    def getSetting(self) -> dict[str, Any]:
        return self.dict_setting

    def getTimestamp(self) -> float:
        return self.s.ts

    def getVolume(self) -> float:
        """出来高"""
        return self.s.volume

    def getVWAP(self) -> float:
        """VWAP"""
        return self.obj_vwap.getValue()

    def isWarmUpPeriod(self) -> float:
        return 1.0 if self.s.step_current < self.PERIOD_WARMUP else 0.0

    def getStepCurrent(self) -> int:
        return self.s.step_current

    def setStepCurrent(self, step: int) -> None:
        self.s.step_current = step

    def setStepCurrentInc(self, incr: int = 1) -> None:
        self.s.step_current += incr

    def getTransaction(self) -> dict[str, list]:
        return self.s.dict_transaction

    def setNHoldInc(self, incr: int = 1) -> None:
        self.s.n_hold += incr

    def setNHoldPositionInc(self, incr: int = 1) -> None:
        self.s.n_hold_position += incr

    def getPnLTotal(self) -> float:
        return self.s.pnl_total

    def addPnLTotal(self, profit: float) -> None:
        self.s.pnl_total += profit

    def getPriceTick(self) -> float:
        return self.s.price_tick

    # ------------------------------------------------------------------
    # ポジション操作系
    # ------------------------------------------------------------------

    def position_close(self) -> float:
        reward = 0.0

        # HOLD カウンター（建玉あり）のリセット
        self.s.n_hold_position = 0
        # 取引回数のインクリメント
        self.s.n_trade += 1

        # 確定損益
        profit = self.getProfit()
        self.s.pnl_total += profit
        reward += profit

        # 取引明細更新（建玉返済）
        if self.s.position != PositionType.NONE:
            self.transaction_close(profit)

        # 含み益関連のリセット
        self.s.price_entry = 0.0
        self.s.profit_max = 0.0
        self.s.drawdown = 0.0
        self.s.dd_ratio = 0.0
        self.s.n_minus = 0

        # ポジションの更新
        self.s.position = PositionType.NONE

        return reward

    def position_open(self, position: PositionType) -> float:
        """新規ポジション"""
        reward = 0.0

        # HOLD カウンター（建玉なし）のリセット
        self.s.n_hold = 0
        # 取引回数のインクリメント
        self.s.n_trade += 1
        # エントリ価格
        self.s.price_entry = self.s.price
        # ポジションを更新
        self.s.position = position
        # 取引明細更新（新規建玉）
        self.transaction_open()

        return reward

    # ------------------------------------------------------------------
    # 設定系
    # ------------------------------------------------------------------

    def setCode(self, code: str) -> None:
        """銘柄コードの設定"""
        self.s.code = code

    # ------------------------------------------------------------------
    # 更新系
    # ------------------------------------------------------------------

    def update(self, ts: float, price: float, volume: float) -> None:
        # --- 最新ティック更新 ---
        self.s.ts = ts
        self.s.price = price
        self.s.volume = volume

        # --- 寄り付き価格の初期化 ---
        if self.s.price_open == 0:
            self.s.price_open = price

        # --- VWAP ---
        vwap = self.obj_vwap.update(price, volume)

        # --- 移動平均 ---
        ma1 = self.obj_ma1.update(price)
        div_ma = ma1 - vwap

        # --- クロス判定 ---
        self.s.cross = self._detect_cross(self.s.div_ma_prev, div_ma)
        self.s.div_ma_prev = div_ma

        # --- ロスカット判定 ---
        self.s.losscut_1 = self.getProfit() <= self.LOSSCUT_1

        # --- 利確判定 ---
        self.s.takeprofit = (
                self.s.dd_ratio > self.DD_RATIO
                and self.s.profit_max > self.DD_PROFIT
        )

    def _detect_cross(self, prev: float | None, curr: float) -> float:
        """移動平均の乖離の符号変化からクロスを検出"""
        if prev is None:
            return 0.0
        if prev < 0 < curr:
            return +1.0
        if curr < 0 < prev:
            return -1.0
        return 0.0

    # ------------------------------------------------------------------
    # 取引明細系
    # ------------------------------------------------------------------

    def transaction_add(self, transaction: str, profit: float = np.nan) -> None:
        """取引明細用データ辞書の更新"""
        self.s.dict_transaction["注文日時"].append(self.get_datetime(self.s.ts))
        self.s.dict_transaction["銘柄コード"].append(self.s.code)
        self.s.dict_transaction["売買"].append(transaction)
        self.s.dict_transaction["約定単価"].append(self.s.price)
        self.s.dict_transaction["約定数量"].append(self.s.unit)
        self.s.dict_transaction["損益"].append(profit)

    def transaction_close(self, profit: float) -> None:
        """建玉返済時の取引明細更新"""
        if self.s.position == PositionType.LONG:
            self.transaction_add("売埋", profit)
        elif self.s.position == PositionType.SHORT:
            self.transaction_add("買埋", profit)
        else:
            raise TypeError(f"Unknown PositionType: {self.s.position}")

    @staticmethod
    def transaction_init() -> dict[str, list]:
        """取引明細用データ辞書の初期化"""
        return _transaction_init()

    def transaction_open(self) -> None:
        """新規建玉時の取引明細更新"""
        if self.s.position == PositionType.LONG:
            self.transaction_add("買建")
        elif self.s.position == PositionType.SHORT:
            self.transaction_add("売建")
        else:
            raise TypeError(f"Unknown PositionType: {self.s.position}")
