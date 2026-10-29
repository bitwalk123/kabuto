from dataclasses import dataclass, field
from typing import Any

import numpy as np

from funcs.conv import position_to_onehot
from modules.technical import (
    EMA,
    MeasureWorm,
    PriceTrendCounter,
    PurePursuitFollower,
    SMA,
    TickATR,
    VWAP,
    WMA,
)
from structs.app_enum import PositionType, CrossStatus


@dataclass
class EnvData:
    # ====== パラメータ ======
    # 約定回数系
    MAX_TRADE: int = 200  # 約定数上限（仮）
    # インジケータ系
    PERIOD_WARMUP: int = 90  # インジケータのウォームアップ期間（ティック数）
    WIDTH_BAND = 5  # バンド幅

    # ====== 実験パラメータ ======
    # 移動平均
    TYPE_MA_1: str = "EMA"
    TYPE_MA_2: str = "EMA"
    PERIOD_MA_1: int = 240  # 移動平均線１の期間
    PERIOD_MA_2: int = 2400  # 移動平均線２の期間
    PERIOD_ATR: int = 60  # ATRの期間
    FLAG_SLOPE: bool = False  # 毎回、傾きの算出をするかどうかのフラグ

    # TickATR を利用したエントリ判定パラメータ
    # THRESHOLD_RATIO: float = 0.0001
    THRESHOLD_RATIO: float = 0  # 0 の時は ATR 無効

    # PPF パラメータ
    GAIN_MA_1: float = 0.005  # PPF の gain
    GAIN_PREDICT_MA_1: float = 0.5  # PPF の gain_predict

    # トレーリング・ストップ
    START_TRAILING: float = 35  # トレーリング・ストップを開始する最大含み益
    THRESHOLD_TRAILING: float = 0.9  # トレーリング・ストップのしきい値比

    # PTC 関連
    FOLLOWING_MIN: int = 14

    # ロスカット系
    LOSSCUT_SIMPLE: float = -50  # 単純ロスカット

    # エントリ関連
    PROFIT_POSITIVE: float = 0.0  # プラ転とみなせるMA含み益
    pend_entry: bool = False  # エントリが保留状態か？

    # ロスカット
    LOSSCUT_VWAP: float = -15  # VWAP基準ロスカット

    # エントリ期間関連
    COUNT_NEGATIVE_MIN: int = 60  # 含み損の許容カウントのしきい値

    PROFIT_THRESHOLD: float = 5  # クロス後、含み益が伸びない判定をするしきい値
    COUNT_NEGATIVE_PROFIT: int = 5  # クロス後、含み益が伸びず含み損を許容できる連続回数

    # インスタンス変数系（初期値が自明な変数のみ）
    row: int = 0  # ティックデータの行位置
    position: PositionType = PositionType.NONE  # ポジション
    position_pre: PositionType = PositionType.NONE  # 一つ前のポジション
    count_negative: int = 0  # エントリの継続カウンタ

    # ティックデータ
    ts: float = 0.0
    price: float = 0.0
    price_cross: float = 0.0  # クロス時の株価

    # クロスの状態
    cross_ma: CrossStatus = CrossStatus.NONE

    # 移動平均
    ma1: float = 0.0
    ma2: float = 0.0
    diff_ma: float = 0.0
    diff_ma_pre: float = 0.0
    slope_ma1: float = 0.0
    slope_ma2: float = 0.0
    # VWAP
    vwap: float = 0.0
    diff_vwap: float = 0.0
    diff_vwap_pre: float = 0.0
    # モメンタム
    mom: float = 0.0
    mom_pre: float = 0.0
    # PriceTrendCounter
    count_high: int = 0  # ローカル高値の継続カウンタ
    count_low: int = 0  # ローカル安値の継続カウンタ
    count_high_pre: int = 0  # ローカル高値の継続カウンタ（前の値）
    count_low_pre: int = 0  # ローカル安値の継続カウンタ（前の値）
    # TickATR
    atr: float = 0.0

    # クロス関連
    ma_gc: bool = False
    ma_dc: bool = False
    vwap_gc: bool = False
    vwap_dc: bool = False

    # 含み損益
    profit: float = 0.0  # 含み損益
    profit_ma: float = 0.0  # 含み損益（移動平均線用）
    profit_max: float = 0.0  # 最大含み損益
    profit_max_ma: float = LOSSCUT_SIMPLE  # 最大含み損益（移動平均線用）
    profit_min_ma: float = 0.0
    profit_pre: float = 0.0  # 一つ前の含み損益
    dd_ratio: float = 0.0  # ドローダウン比率

    # 建玉返済ロジック
    status_cross_ma: bool = False
    status_cross_vwap: bool = False
    status_cross_vwap_2: bool = False
    status_profit_trailing: bool = False
    status_losscut_vwap: bool = False
    status_threshold: bool = False

    # ====== マスク処理関連 ======
    MASK_HOLD_ONLY = np.array([True, False, False], dtype=np.bool_)
    # 取りうるアクション: HOLD, BUY, SELL
    MASK_ALL = np.array([True, True, True], dtype=np.bool_)
    # 取りうるアクション: HOLD, SELL
    MASK_LONG = np.array([True, False, True], dtype=np.bool_)
    # 取りうるアクション: HOLD, BUY
    MASK_SHORT = np.array([True, True, False], dtype=np.bool_)

    POSITION_MASKS = {
        # 建玉なし
        PositionType.NONE: MASK_ALL,
        # LONG
        PositionType.LONG: MASK_LONG,
        # SHORT
        PositionType.SHORT: MASK_SHORT,
    }

    # テクニカル指標のインスタンス
    obj_ma_1: Any = field(init=False)
    obj_ma_2: Any = field(init=False)
    obj_vwap: VWAP = field(init=False)
    obj_ptc: PriceTrendCounter = field(init=False)
    obj_atr: TickATR = field(init=False)

    def __post_init__(self):
        self.bands_golden = [
            self.WIDTH_BAND,
        ]
        self.bands_dead = [
            -self.WIDTH_BAND,
        ]
        # テクニカル指標のインスタンスの初期化
        if self.TYPE_MA_1 == "PPF":
            self.obj_ma_1 = PurePursuitFollower(
                self.PERIOD_MA_1,
                self.GAIN_MA_1,
                self.GAIN_PREDICT_MA_1,
            )
        elif self.TYPE_MA_1 == "SMA":
            self.obj_ma_1 = SMA(self.PERIOD_MA_1)
        elif self.TYPE_MA_1 == "EMA":
            self.obj_ma_1 = EMA(self.PERIOD_MA_1)
        elif self.TYPE_MA_1 == "WMA":
            self.obj_ma_1 = WMA(self.PERIOD_MA_1)
        elif self.TYPE_MA_1 == "MW":
            self.obj_ma_1 = MeasureWorm(self.PERIOD_MA_1)
        else:
            raise TypeError("TYPE_MA_1 に合致しません。")

        if self.TYPE_MA_2 == "SMA":
            self.obj_ma_2 = SMA(self.PERIOD_MA_2)
        elif self.TYPE_MA_2 == "EMA":
            self.obj_ma_2 = EMA(self.PERIOD_MA_2)
        else:
            raise TypeError("TYPE_MA_2 に合致しません。")

        self.obj_vwap = VWAP()
        self.obj_ptc = PriceTrendCounter()
        self.obj_atr = TickATR(self.PERIOD_ATR)

    def print_param(self):
        # ====== パラメータ ======
        # 約定回数系
        # インジケータ系
        print("PERIOD_WARMUP", self.PERIOD_WARMUP)  # インジケータのウォームアップ期間（ティック数）
        print("PERIOD_MA_2", self.PERIOD_MA_2)  # 移動平均線の期間2

    def inc_row(self):
        self.row += 1

    def get_masks(self):
        """
        行動マスク
        【マスク】
        - ウォーミングアップ期間 → 強制 HOLD
        - ナンピン取引の禁止

        :return: mask
        """
        if self.row < self.PERIOD_WARMUP:
            # ウォーミングアップ期間 → 強制 HOLD
            return self.MASK_HOLD_ONLY

        try:
            return self.POSITION_MASKS[self.position]
        except KeyError:
            raise TypeError(f"Unknown PositionType: {self.position}")

    def get_obs(self) -> dict:
        """
        観測空間の算出
        :return:
        """
        # シグナル・フラグ（クロスしたタイミングなど）
        if not self.is_first_entry_timing():
            self.ma_gc = self.is_ma_golden_cross()
            self.ma_dc = self.is_ma_dead_cross()
        else:
            # WARMUP 後の最初のエントリ・タイミング
            self.ma_gc, self.ma_dc = self.try_first_entry()

        self.vwap_gc = self.is_vwap_golden_cross()
        self.vwap_dc = self.is_vwap_dead_cross()

        signal = np.array([
            self.ma_gc,  # 0. MA ゴールデンクロスのフラグ
            self.ma_dc,  # 1. MA デッドクロスのフラグ
            self.vwap_gc,  # 2. VWAP ゴールデンクロスのフラグ
            self.vwap_dc,  # 3. VWAP デッドクロスのフラグ
            False,  # 5. 予備
            self.does_take_profit(),  # 5. 利確のフラグ
            self.judge_ptc(),  # 6. PTC判定
            self.is_losscut(),  # 7. 単純ロスカットのフラグ
            False,
            False,
        ])
        # ポジション情報
        position_onehot = position_to_onehot(self.position)

        # 辞書形式で返す
        return {
            "signal": signal,
            "position": position_onehot,
        }

    def get_technicals(self) -> dict:
        return {
            "ts": self.ts,
            "price": self.price,
            "ma1": self.ma1,
            "ma2": self.ma2,
            "slope_ma1": self.slope_ma1,
            "slope_ma2": self.slope_ma2,
            "vwap": self.vwap,
            "atr": self.atr,
            "profit": self.profit,
            "profit_ma": self.profit_ma,
            "profit_max": self.profit_max,
            "profit_max_ma": self.profit_max_ma,
            "profit_min_ma": self.profit_min_ma,
            "dd_ratio": self.dd_ratio,
            "diff_ma": self.diff_ma,
            "diff_vwap": self.diff_vwap,
            "count_negative": self.count_negative,
            "count_high": self.count_high,
            "count_low": self.count_low,
            "count_high_pre": self.count_high_pre,
            "count_low_pre": self.count_low_pre,
            "ma_gc": self.ma_gc,
            "ma_dc": self.ma_dc,
            "vwap_gc": self.vwap_gc,
            "vwap_dc": self.vwap_dc,
            "warmup": self.is_warmup_period(),
        }

    def try_first_entry(self) -> tuple[bool, bool]:
        if not self.status_cross_ma:
            # 「クロス MA エントリ/返済」が有効になっていない場合
            return False, False

        if self.ma2 < self.ma1 < self.price:
            # MA ゴールデン・クロスとして扱う
            return True, False

        if self.price < self.ma1 < self.ma2:
            # MA デッド・クロスとして扱う
            return False, True

        return False, False

    def is_ma_golden_cross(self) -> bool:
        """MA ゴールデン・クロスか？"""
        return self.check_ma_cross(CrossStatus.GOLDEN)

    def is_ma_dead_cross(self) -> bool:
        """MA デッド・クロスか？"""
        return self.check_ma_cross(CrossStatus.DEAD)

    def check_ma_cross(self, cross: CrossStatus) -> bool:
        if not self.status_cross_ma:
            # 「クロス MA エントリ/返済」が有効になっていない場合
            return False

        # 指定方向でクロスしているかどうか？
        crossed = self.is_crossed(cross)
        if self.THRESHOLD_RATIO > 0:
            # THRESHOLD_RATIO が 0 より大きい場合は ATR 有効
            if self.position == PositionType.NONE:
                # 建玉がない場合の ATR によるエントリ判定
                return self.check_atr_entry(crossed)
            else:
                # 建玉がある場合
                return crossed
        else:
            # THRESHOLD_RATIO が 0 の場合は ATR 無効
            return crossed

    def check_atr_entry(self, crossed: bool) -> bool:
        if crossed:
            if self.atr < self.price * self.THRESHOLD_RATIO:
                # エントリ見送り
                print("エントリが見送られました。")
                return False
            else:
                return True
        else:
            return False

    def is_crossed(self, cross: CrossStatus) -> bool:
        """指定方向のクロスが発生したか"""
        if cross == CrossStatus.GOLDEN:
            return self.diff_ma_pre <= 0 < self.diff_ma
        return self.diff_ma < 0 <= self.diff_ma_pre

    def calc_profit_ma(self, cross: CrossStatus) -> float:
        """クロス時の価格を基準にした、MA1 までの見込み利益"""
        if cross == CrossStatus.GOLDEN:
            return self.ma1 - self.price_cross
        return self.price_cross - self.ma1

    def is_vwap_golden_cross(self) -> bool:
        """
        VWAP ゴールデン・クロスでエントリか？
        :return:
        """
        if self.status_cross_vwap:
            if self.diff_vwap_pre <= 0 < self.diff_vwap:
                return True

        if self.status_cross_vwap_2:
            for v in self.bands_golden:
                if self.diff_vwap_pre <= v < self.diff_vwap:
                    return True

        return False

    def is_vwap_dead_cross(self) -> bool:
        """
        VWAP デッド・クロスでエントリか？
        :return:
        """
        if self.status_cross_vwap:
            if self.diff_vwap < 0 <= self.diff_vwap_pre:
                return True

        if self.status_cross_vwap_2:
            for v in self.bands_dead:
                if self.diff_vwap < v <= self.diff_vwap_pre:
                    return True

        return False

    def is_warmup_period(self) -> float:
        return 1.0 if self.row < self.PERIOD_WARMUP else 0.0

    def is_first_entry_timing(self) -> bool:
        """
        self.PERIOD_WARMUP後の最初の可能なエントリ・タイミングか？
        :return:
        """
        return True if self.row == self.PERIOD_WARMUP + 1 else False

    def reset_count_entry(self):
        self.count_negative = 0

    def reset_profit_pre(self):
        self.profit_pre = 0.0

    def set_data(self, ts: float, price: float, volume: float, dict_info: dict):
        self.ts = ts
        self.price = price
        self.position = dict_info["position"]

        self.atr = atr = self.obj_atr.update(price)  # TickATR
        if self.TYPE_MA_1 == "MW":
            self.ma1 = ma1 = self.obj_ma_1.update(price, atr)
        else:
            self.ma1 = ma1 = self.obj_ma_1.update(price)
        self.ma2 = ma2 = self.obj_ma_2.update(price)
        self.diff_ma = ma1 - ma2
        self.vwap = vwap = self.obj_vwap.update(price, volume)
        self.diff_vwap = ma1 - vwap

        if self.FLAG_SLOPE:
            # 常時、傾きを算出する場合
            self.slope_ma1 = self.obj_ma_1.getSlope()
            self.slope_ma2 = self.obj_ma_2.getSlope()

        self.count_high, self.count_low = self.obj_ptc.update(self.position, price, ma1, ma2)

        self.profit = dict_info["profit"]
        # 移動平均線ベースの含み益
        # dict_info["price"]: 建玉の株価
        if self.position == PositionType.LONG:
            self.profit_ma = ma1 - dict_info["price"]
        elif self.position == PositionType.SHORT:
            self.profit_ma = dict_info["price"] - ma1
        else:
            self.profit_ma = 0

        self.update_profit_max()  # 含み損益の最大値を更新
        self.update_count_negative()  # 含み損の継続カウンタの更新

        obs = self.get_obs()  # 【重要】必ず観測値の取得が先
        dict_technical = self.get_technicals()

        # 一つ前の特徴量の更新
        self.update_feature_pre()

        # ステップ（データフレームの行）更新
        self.inc_row()

        return obs, dict_technical

    def setStatusCrossMA(self, state: bool) -> bool:
        self.status_cross_ma = state
        return self.status_cross_ma

    def setStatusCrossVWAP(self, state: bool) -> bool:
        self.status_cross_vwap = state
        return self.status_cross_vwap

    def setStatusProfitTrailing(self, state: bool) -> bool:
        self.status_profit_trailing = state
        return self.status_profit_trailing

    def setStatusLosscutVWAP(self, state: bool) -> bool:
        self.status_losscut_vwap = state
        return self.status_losscut_vwap

    def setStatusThreshold(self, state: bool) -> bool:
        self.status_threshold = state
        return self.status_threshold

    def update_count_negative(self):
        if self.position != PositionType.NONE:
            if self.profit <= 0:
                # 含み損であればカウンタを更新
                self.count_negative += 1
            else:
                # 含み益であればリセット
                self.count_negative = 0
        else:
            self.count_negative = 0

    def update_feature_pre(self):
        self.diff_ma_pre = self.diff_ma
        self.diff_vwap_pre = self.diff_vwap
        self.count_high_pre = self.count_high
        self.count_low_pre = self.count_low

        if self.position == PositionType.NONE:
            self.dd_ratio = 0.0
            self.count_negative = 0
            self.profit_max = 0.0
            self.profit_max_ma = self.LOSSCUT_SIMPLE
            self.profit_min_ma = 0.0

        self.profit_pre = self.profit

    def reset_profit_max(self):
        self.profit_max = 0.0
        self.profit_max_ma = self.LOSSCUT_SIMPLE
        self.profit_min_ma = 0.0

    def update_profit_max(self):
        """
        含み損益の最大値を更新
        :return:
        """
        if self.profit_max < self.profit:
            self.profit_max = self.profit
        if self.profit_max_ma < self.profit_ma:
            self.profit_max_ma = self.profit_ma
        if self.profit_ma < self.profit_min_ma:
            self.profit_min_ma = self.profit_ma

    def is_losscut(self) -> bool:
        if self.status_losscut_vwap:
            # VWAP基準のロスカット
            if self.position == PositionType.LONG:
                if self.price - self.vwap <= self.LOSSCUT_VWAP:
                    return True
            if self.position == PositionType.SHORT:
                if self.vwap - self.price <= self.LOSSCUT_VWAP:
                    return True

        # if 0 < self.profit_max < self.START_TRAILING and self.profit_ma < -20:
        #    return True

        if self.profit < self.LOSSCUT_SIMPLE:
            # 単純ロスカット
            return True

        return False

    def does_take_profit(self) -> bool:
        if self.status_profit_trailing:
            if self.START_TRAILING <= self.profit_max_ma:
                if self.profit_ma < self.profit_max_ma * self.THRESHOLD_TRAILING:
                    return True
        return False

    def update_profit_pre(self):
        self.profit_pre = self.profit  # 一つ前の含み益の更新

    def judge_ptc(self) -> bool:
        """
        PTC 判定
        :return:
        """
        return False  # 現時点では無効化
        if not self.status_cross_ma:
            # 「クロス MA エントリ/返済」が有効になっていない場合
            # PTC 判定は MA クロスとセット
            return False

        if self.position == PositionType.NONE:
            return False
        if self.position == PositionType.LONG:
            # Long
            following = self.count_high
            contrarian = self.count_low
        else:
            # Short
            following = self.count_low
            contrarian = self.count_high

        # 順行回数が一定回数以上
        if self.FOLLOWING_MIN <= following:
            # == ロスカット
            # 逆行回数が順行回数より多い
            if following <= contrarian:
                # 含み益が一定以下
                if self.profit <= 0:
                    return True

        return False
