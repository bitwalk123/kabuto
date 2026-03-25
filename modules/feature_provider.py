import datetime
from dataclasses import dataclass, field
from typing import Any

from funcs.commons import detect_cross, init_transaction, detect_cross_golden, detect_cross_dead
from modules.technical import MovingAverage, VWAP, RSI, Momentum
from structs.defaults import FeatureDefaults
from structs.app_enum import PositionType


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

    # 移動平均 & VWAP 関連
    div_ma_prev: float | None = None
    div_ma2_prev: float | None = None
    div_ma3_prev: float | None = None

    # クロス・シグナル
    cross_1: float = 0.0  # VWAP と MA1 のクロス・シグナル
    cross_2: float = 0.0  # VWAP 上バンドと MA1 のゴールデン・クロス
    cross_3: float = 0.0  # VWAP 下バンドと MA1 のデッド・クロス

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
    dict_transaction: dict[str, list] = field(default_factory=init_transaction)
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

    def __init__(self, dict_setting: dict[str, Any]) -> None:
        # 1. 最初に dict_setting を初期化
        self.dict_setting: dict[str, Any] = self._merge_with_defaults(dict_setting)
        print("パラメータ")
        for key, value in self.dict_setting.items():
            print(f"{key} : {value}")

        # 2. 固定パラメータを使ってインスタンス生成
        self.N_TRADE_MAX: int = 100
        self.obj_vwap = VWAP()
        self.obj_ma1 = MovingAverage(window_size=self.dict_setting["PERIOD_MA_1"])
        self.obj_rsi = RSI(window_size=self.dict_setting["PERIOD_RSI"])
        self.obj_mom = Momentum(window_size=self.dict_setting["PERIOD_MOM"])

        self.s: FeatureState = FeatureState()
        self.clear()

    @staticmethod
    def _merge_with_defaults(dict_setting: dict[str, Any]) -> dict[str, Any]:
        """
        デフォルト値とマージするヘルパー

        :param dict_setting:
        :return:
        """
        defaults = FeatureDefaults.as_dict()
        return {**defaults, **dict_setting}

    def clear(self) -> None:
        # オブジェクト系は個別に clear()
        self.obj_vwap.clear()
        self.obj_ma1.clear()
        self.obj_rsi.clear()
        self.obj_mom.clear()

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
        """クロスシグナル1（-1.0, 0.0, 1.0）"""
        return float(self.s.cross_1)

    def getCrossSignal2(self) -> float:
        """クロスシグナル2（0.0, 1.0）"""
        return float(self.s.cross_2)

    def getCrossSignal3(self) -> float:
        """クロスシグナル3（-1.0, 0.0）"""
        return float(self.s.cross_3)

    def getCurrentPosition(self) -> PositionType:
        return self.s.position

    def getDDRatio(self) -> float:
        """ドローダウン比率"""
        return self.s.dd_ratio

    def getDrawDown(self) -> float:
        """ドローダウン"""
        return self.s.drawdown

    def getLosscut1(self) -> float:
        return 1.0 if self.s.losscut_1 else 0.0

    def getLosscut2(self) -> float:
        return 1.0 if self.s.n_minus > self.dict_setting["N_MINUS_MAX"] else 0.0

    def getMA1(self) -> float:
        """移動平均 1 の取得"""
        return self.obj_ma1.getValue()

    def getRSI(self) -> float:
        """RSI の取得"""
        return self.obj_rsi.getValue()

    def getMOM(self) -> float:
        """Momentum の取得"""
        return self.obj_mom.getValue()

    def getNTrade(self) -> int:
        return self.s.n_trade

    def getPeriodWarmup(self) -> int:
        return self.dict_setting["PERIOD_WARMUP"]

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
        return 1.0 if self.s.step_current < self.dict_setting["PERIOD_WARMUP"] else 0.0

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

        return reward

    # ------------------------------------------------------------------
    # 設定系
    # ------------------------------------------------------------------

    def setCode(self, code: str) -> None:
        """銘柄コードの設定"""
        self.s.code = code

    def updateSetting(self, dict_setting: dict[str, Any]) -> None:
        """
        実行時に動的設定を変更。
        注意: PERIOD_MA_1 などを変更しても obj_ma1 は再生成されない。

        :param dict_setting:
        :return:
        """
        # 変更禁止キーのチェック
        frozen_keys = {"PERIOD_MA_1", "PERIOD_WARMUP"}

        # 変更可能なキーのみ抽出
        updatable_settings = {}
        for key, value in dict_setting.items():
            if key in frozen_keys:
                print(f"  {key} は変更できません（スキップします）")
            else:
                updatable_settings[key] = value

        # 変更可能な設定のみ更新
        self.dict_setting.update(updatable_settings)

        print("設定を更新しました")
        for key, value in updatable_settings.items():
            print(f"  {key} : {value}")

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
        rsi = self.obj_rsi.update(ma1)
        mom = self.obj_mom.update(ma1)
        div_ma = ma1 - vwap
        delta = self.dict_setting["BAND_VWAP"]
        div_ma2 = ma1 - vwap - delta
        div_ma3 = ma1 - vwap + delta

        # --- クロス判定 ---
        self.s.cross_1 = detect_cross(self.s.div_ma_prev, div_ma)
        self.s.cross_2 = detect_cross_golden(self.s.div_ma2_prev, div_ma2)
        self.s.cross_3 = detect_cross_dead(self.s.div_ma3_prev, div_ma3)
        self.s.div_ma_prev = div_ma
        self.s.div_ma2_prev = div_ma2
        self.s.div_ma3_prev = div_ma3

        # --- ロスカット判定 ---
        self.s.losscut_1 = self.getProfit() <= self.dict_setting["LOSSCUT_1"]

        # --- 利確判定 ---
        self.s.takeprofit = (
                self.dict_setting["DD_RATIO"] < self.s.dd_ratio
                and self.dict_setting["DD_PROFIT"] < self.s.profit_max
        )
