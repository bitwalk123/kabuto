import datetime
from collections import deque
from enum import Enum
from typing import override

import gymnasium as gym
import numpy as np
import pandas as pd
import talib


class ActionType(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2
    REPAY = 3


class PositionType(Enum):
    NONE = 0
    LONG = 1
    SHORT = 2


class FeatureProvider:
    def __init__(self):
        self.ts = 0
        self.price = 0
        self.volume = 0
        self.vwap = 0

        # 特徴量算出のために保持する変数
        self.price_open = 0.0  # ザラバの始値
        self.cum_pv = 0.0  # VWAP 用 Price × Volume 累積
        self.cum_vol = 0.0  # VWAP 用 Volume 累積
        self.volume_prev = None  # VWAP 用 前の Volume

        # キューを定義
        self.n_deque_price = 300
        self.deque_price = deque(maxlen=self.n_deque_price)  # 移動平均など

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

        # 特徴量算出のために保持する変数
        self.price_open = 0.0  # ザラバの始値
        self.cum_pv = 0.0  # VWAP 用 Price × Volume 累積
        self.cum_vol = 0.0  # VWAP 用 Volume 累積
        self.volume_prev = None  # VWAP 用 前の Volume

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

    def getPriceRatio(self) -> float:
        """
        （始値で割った）株価比
        """
        return self.price / self.price_open if self.price_open > 0 else 0.0

    def getRSI(self) -> float:
        """
        VWAP 乖離率 (deviation rate = dr)
        """
        n = len(self.deque_price)
        if n > 2:
            array_rsi = talib.RSI(
                np.array(self.deque_price, dtype=np.float64),
                timeperiod=n - 1
            )
            return array_rsi[-1]
        else:
            return 0.

    def getVWAPdr(self) -> float:
        if self.vwap == 0.0:
            return 0.0
        else:
            return (self.price - self.vwap) / self.vwap

    def update(self, ts, price, volume):
        # 最新ティック情報を保持
        self.ts = ts
        if self.price_open == 0.0:
            """
            寄り付いた最初の株価が基準価格
            ※ 寄り付き後の株価が送られてくることをシステムが保証している
            """
            self.price_open = price
        self.price = price
        self.volume = volume
        self.vwap = self._calc_vwap()

        # キューへの追加
        self.deque_price.append(price)
        # self.deque_dvol_002.append(volume)


class TransactionManager:
    """
    売買管理クラス
    方策マスクでナンピンをしないことが前提
    """

    def __init__(self, provider: FeatureProvider, code: str = '7011'):
        # 特徴量プロバイダ
        self.provider = provider
        self.code: str = code  # 銘柄コード
        # ---------------------------------------------------------------------
        # 取引関連
        # ---------------------------------------------------------------------
        self.unit: int = 1  # 売買単位
        self.tickprice: float = 1.0  # 呼び値
        self.position = PositionType.NONE  # ポジション（建玉）
        self.price_entry = 0.0  # 取得価格
        self.pnl_total = 0.0  # 総損益
        self.dict_transaction = self.init_transaction()  # 取引明細
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # 報酬設計
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # 含み損益の場合に乗ずる比率
        self.ratio_unreal_profit = 0.1
        # 含み損益の保持のカウンター
        self.count_unreal_profit_weighted = 0
        # 含み損益のインセンティブ・ペナルティ比率
        self.ratio_unreal_profit_weighted = 0.5
        # 報酬の平方根処理で割る因子
        self.factor_reward_sqrt = 25.0
        # エントリ時のVWAP に紐づく報酬ファクター
        self.factor_vwap_scaling = 0.0

    def add_transaction(self, transaction: str, profit: float = np.nan):
        self.dict_transaction["注文日時"].append(self.get_datetime(self.provider.ts))
        self.dict_transaction["銘柄コード"].append(self.code)
        self.dict_transaction["売買"].append(transaction)
        self.dict_transaction["約定単価"].append(self.provider.price)
        self.dict_transaction["約定数量"].append(self.unit)
        self.dict_transaction["損益"].append(profit)

    def clear(self):
        self.clear_position()
        self.pnl_total = 0.0  # 総損益
        self.dict_transaction = self.init_transaction()  # 取引明細

    def clear_position(self):
        self.position = PositionType.NONE
        self.price_entry = 0.0
        self.count_unreal_profit_weighted = 0

    def evalReward(self, action: int) -> float:
        action_type = ActionType(action)
        reward = 0.0
        if self.position == PositionType.NONE:
            # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
            # ポジションが無い場合に取りうるアクションは HOLD, BUY, SELL
            # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
            if action_type == ActionType.HOLD:
                pass
            elif action_type == ActionType.BUY:
                # =============================================================
                # 買建 (LONG)
                # =============================================================
                self.position = PositionType.LONG  # ポジションを更新
                self.price_entry = self.provider.price  # 取得価格
                reward += np.tanh(
                    (self.provider.vwap - self.price_entry) / self.provider.vwap * self.factor_vwap_scaling)
                # -------------------------------------------------------------
                # 取引明細
                # -------------------------------------------------------------
                self.add_transaction("買建")
            elif action_type == ActionType.SELL:
                # =============================================================
                # 売建 (SHORT)
                # =============================================================
                self.position = PositionType.SHORT  # ポジションを更新
                self.price_entry = self.provider.price  # 取得価格
                reward += np.tanh(
                    (self.price_entry - self.provider.vwap) / self.provider.vwap * self.factor_vwap_scaling)
                # -------------------------------------------------------------
                # 取引明細
                # -------------------------------------------------------------
                self.add_transaction("売建")
            elif action_type == ActionType.REPAY:
                # 取引ルール違反
                raise TypeError(f"Violation of transaction rule: {action_type}")
            else:
                raise TypeError(f"Unknown ActionType: {action_type}")
        else:
            # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
            # ポジションが有る場合に取りうるアクションは HOLD, REPAY
            # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
            if action_type == ActionType.HOLD:
                # =============================================================
                # 含み益
                # =============================================================
                profit = self.get_profit()
                # 含み益を持ち続けることで付与されるボーナス
                self.count_unreal_profit_weighted += 1
                k = self.count_unreal_profit_weighted * self.ratio_unreal_profit_weighted
                profit_weighted = profit * (1 + k)
                reward += self.get_reward_from_profit(profit_weighted) * self.ratio_unreal_profit
            elif action_type == ActionType.BUY:
                # 取引ルール違反
                raise TypeError(f"Violation of transaction rule: {action_type}")
            elif action_type == ActionType.SELL:
                # 取引ルール違反
                raise TypeError(f"Violation of transaction rule: {action_type}")
            elif action_type == ActionType.REPAY:
                # =============================================================
                # 返済
                # =============================================================
                profit = self.get_profit()
                # 損益追加
                self.pnl_total += profit
                # 報酬
                reward += self.get_reward_from_profit(profit)
                # -------------------------------------------------------------
                # 取引明細
                # -------------------------------------------------------------
                if self.position == PositionType.LONG:
                    # 返済: 買建 (LONG) → 売埋
                    self.add_transaction("売埋", profit)
                elif self.position == PositionType.SHORT:
                    # 返済: 売建 (SHORT) → 買埋
                    self.add_transaction("買埋", profit)
                else:
                    raise TypeError(f"Unknown PositionType: {self.position}")
                # =============================================================
                # ポジション解消
                # =============================================================
                self.clear_position()
            else:
                raise TypeError(f"Unknown ActionType: {action_type}")

        return reward

    def forceRepay(self) -> float:
        reward = 0.0
        profit = self.get_profit()
        if self.position == PositionType.LONG:
            # 返済: 買建 (LONG) → 売埋
            # -------------------------------------------------------------
            # 取引明細
            # -------------------------------------------------------------
            self.add_transaction("売埋（強制返済）", profit)
        elif self.position == PositionType.SHORT:
            # 返済: 売建 (SHORT) → 買埋
            # -------------------------------------------------------------
            # 取引明細
            # -------------------------------------------------------------
            self.add_transaction("買埋（強制返済）", profit)
        else:
            # ポジション無し
            pass
        # 損益追加
        self.pnl_total += profit
        # 報酬
        reward += self.get_reward_from_profit(profit)
        # =====================================================================
        # ポジション解消
        # =====================================================================
        self.clear_position()

        return reward

    @staticmethod
    def get_datetime(t: float) -> str:
        return str(datetime.datetime.fromtimestamp(int(t)))

    def get_profit(self) -> float:
        if self.position == PositionType.LONG:
            # ---------------------------------------------------------
            # 返済: 買建 (LONG) → 売埋
            # ---------------------------------------------------------
            return self.provider.price - self.price_entry
        elif self.position == PositionType.SHORT:
            # ---------------------------------------------------------
            # 返済: 売建 (SHORT) → 買埋
            # ---------------------------------------------------------
            return self.price_entry - self.provider.price
        else:
            return 0.0  # 実現損益

    def get_reward_from_profit(self, profit: float) -> float:
        # 報酬は呼び値で割る
        return np.sign(profit) * np.sqrt(abs(profit / self.tickprice)) / self.factor_reward_sqrt

    def getNumberOfTransactions(self) -> int:
        return len(self.dict_transaction["注文日時"])

    def getPL4Obs(self) -> float:
        """
        観測値用に、損益用の報酬と同じにスケーリングして含み損益を返す。
        """
        profit = self.get_profit()
        return self.get_reward_from_profit(profit)

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


class ObservationManager:
    def __init__(self, provider: FeatureProvider):
        # 特徴量プロバイダ
        self.provider = provider
        # 調整用係数
        self.tickprice = 1.0  # 呼び値
        self.unit = 100  # 最小取引単位（出来高）
        self.factor_hold = 10_000.  # 建玉保持カウンタ用
        self.factor_ma_diff = 0.05  # 移動平均差用
        self.factor_price = 20.  # 株価用
        self.factor_vwap = 30.0  # VWAP用
        """
        観測量（特徴量）数の取得
        観測量の数 (self.n_feature) は、評価によって頻繁に変動するので、
        コンストラクタでダミー（空）を実行して数を自律的に把握できるようにする。
        """
        self.n_feature = len(self.getObs())
        self.clear()  # ダミーを実行したのでリセット

    def clear(self):
        self.provider.clear()

    def func_ma_scaling(self, ma: float) -> float:
        if self.provider.price_open == 0.0:
            ma_ratio = 0.0
        else:
            ma_ratio = (ma / self.provider.price_open - 1.0) * self.factor_price
        return ma_ratio

    def func_ma_diff_scaling(self, ma_diff: float) -> float:
        if self.provider.price_open == 0.0:
            ma_diff_scaled = 0.0
        else:
            ma_diff_scaled = ma_diff / self.tickprice  * self.factor_ma_diff
        return np.tanh(ma_diff_scaled)

    def getObs(
            self,
            pl: float = 0,  # 含み損益
            count_hold: int = 0,  # HOLD 継続カウンタ
            position: PositionType = PositionType.NONE  # ポジション
    ) -> np.ndarray:
        # 観測値（特徴量）用リスト
        list_feature = list()
        # ---------------------------------------------------------------------
        # 1. 株価比率
        # ---------------------------------------------------------------------
        price_ratio = self.provider.getPriceRatio()
        price_ratio = (price_ratio - 1.0) * self.factor_price
        list_feature.append(price_ratio)
        # 移動平均
        ma_060 = self.provider.getMA(60)
        ma_120 = self.provider.getMA(120)
        ma_300 = self.provider.getMA(300)
        # ---------------------------------------------------------------------
        # 2. MAΔ1: 移動平均の差分 MA60 - MA120
        # ---------------------------------------------------------------------
        # 移動平均の算出 1
        ma_diff_1 = ma_060 - ma_120
        list_feature.append(self.func_ma_diff_scaling(ma_diff_1))
        # ---------------------------------------------------------------------
        # 3. MAΔ2: 移動平均の差分 MA60 - MA300
        # ---------------------------------------------------------------------
        # 移動平均の算出 2
        ma_diff_2 = ma_060 - ma_300
        list_feature.append(self.func_ma_diff_scaling(ma_diff_2))
        # ---------------------------------------------------------------------
        # 4. RSI: [-1, 1] に標準化
        # ---------------------------------------------------------------------
        rsi = self.provider.getRSI()
        rsi_scaled = (rsi - 50.) / 50.
        list_feature.append(rsi_scaled)
        # ---------------------------------------------------------------------
        # 5. VWAP 乖離率 (deviation rate = dr)
        # ---------------------------------------------------------------------
        vwap_dr = self.provider.getVWAPdr()
        vwap_dr_scaled = np.clip(vwap_dr * self.factor_vwap, -1.0, 1.0)
        list_feature.append(vwap_dr_scaled)
        # ---------------------------------------------------------------------
        # 6. 含み損益
        # ---------------------------------------------------------------------
        list_feature.append(pl)
        # ---------------------------------------------------------------------
        # 7. HOLD 継続カウンタ
        # ---------------------------------------------------------------------
        list_feature.append(np.tanh(count_hold / self.factor_hold))
        # 一旦、配列に変換
        arr_feature = np.array(list_feature, dtype=np.float32)
        # ---------------------------------------------------------------------
        # ポジション情報
        # 8., 9., 10. PositionType → one-hot (3) ［単位行列へ変換］
        # ---------------------------------------------------------------------
        pos_onehot = np.eye(len(PositionType))[position.value].astype(np.float32)
        # arr_feature と pos_onehot を単純結合
        return np.concatenate([arr_feature, pos_onehot])

    def getObsReset(self) -> np.ndarray:
        obs = self.getObs()
        self.clear()
        return obs


class TradingEnv(gym.Env):
    # 環境クラス
    def __init__(self):
        super().__init__()
        # ウォームアップ期間
        self.n_warmup: int = 60
        # 現在の行位置
        self.step_current: int = 0
        # 特徴量プロバイダ
        self.provider = provider = FeatureProvider()
        # 売買管理クラス
        self.trans_man = TransactionManager(provider)
        # 観測値管理クラス
        self.obs_man = ObservationManager(provider)
        # 観測空間
        n_feature = self.obs_man.n_feature
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_feature,),
            dtype=np.float32
        )
        # 行動空間
        self.action_space = gym.spaces.Discrete(len(ActionType))

    def _get_action_mask(self) -> np.ndarray:
        # 行動マスク
        if self.step_current < self.n_warmup:
            """
            ウォーミングアップ期間
            強制 HOLD
            """
            return np.array([1, 0, 0, 0], dtype=np.int8)
        elif self.trans_man.position == PositionType.NONE:
            """
            建玉なし
            取りうるアクション: HOLD, BUY, SELL
            """
            return np.array([1, 1, 1, 0], dtype=np.int8)
        else:
            """
            建玉あり
            取りうるアクション: HOLD, REPAY
            """
            return np.array([1, 0, 0, 1], dtype=np.int8)

    def _get_tick(self) -> tuple[float, float, float]:
        ...

    def getTransaction(self) -> pd.DataFrame:
        return pd.DataFrame(self.trans_man.dict_transaction)

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        self.step_current = 0
        self.trans_man.clear()
        obs = self.obs_man.getObsReset()
        return obs, {"action_mask": self._get_action_mask()}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        ...


class TrainingEnv(TradingEnv):
    """
    環境クラス
    過去のティックデータを使った学習、推論用
    """

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df.reset_index(drop=True)  # Time, Price, Volume のみ

    @override
    def _get_tick(self) -> tuple[float, float, float]:
        t: float = self.df.at[self.step_current, "Time"]
        price: float = self.df.at[self.step_current, "Price"]
        volume: float = self.df.at[self.step_current, "Volume"]
        return t, price, volume

    @override
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        過去のティックデータを使うことを前提とした step 処理
        """
        # --- ウォームアップ期間 (self.n_warmup) は強制 HOLD ---
        if self.step_current < self.n_warmup:
            action = ActionType.HOLD.value

        # データフレームからティックデータを取得
        # t, price, volume = self._get_tick()
        self.provider.update(*self._get_tick())
        # 報酬
        reward = self.trans_man.evalReward(action)
        # 観測値
        obs = self.obs_man.getObs(
            self.trans_man.getPL4Obs(),  # 含み損益
            self.trans_man.count_unreal_profit_weighted,  # HOLD 継続カウンタ
            self.trans_man.position,  # ポジション
        )

        done = False
        truncated = False

        if self.step_current >= len(self.df) - 1:
            reward += self.trans_man.forceRepay()
            done = True
            truncated = True  # ← 時間切れによる終了を明示

        self.step_current += 1
        info = {
            "pnl_total": self.trans_man.pnl_total,
            "action_mask": self._get_action_mask()
        }

        return obs, reward, done, truncated, info
