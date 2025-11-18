import datetime

import numpy as np

from modules.provider import FeatureProvider
from structs.app_enum import PositionType, ActionType


class RewardManager:
    """
    アクションに対して報酬の支払うクラス
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
        self.profit_max = 0.0  # 含み損益の最大値
        self.dict_transaction = self.init_transaction()  # 取引明細
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # 報酬設計
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # 含み損益のインセンティブ・ペナルティ比率
        self.ratio_hold_position = 0.5
        # 含み損益の場合に乗ずる比率
        self.ratio_unreal_profit = 0.075
        # 報酬の平方根処理で割る因子
        self.factor_reward_sqrt = 25.0
        # エントリ時のVWAP に紐づく報酬ファクター
        self.factor_vwap_scaling = 0.0
        # 取引コストペナルティ
        self.penalty_trade_count = 0.01
        # 建玉なしで僅かな報酬・ペナルティ
        self.reward_penalty_hold = +0.00001

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
        self.provider.resetTradeCounter()  # 取引回数カウンターのリセット
        self.dict_transaction = self.init_transaction()  # 取引明細

    def clear_position(self):
        self.position = PositionType.NONE
        self.price_entry = 0.0
        self.profit_max = 0.0  # 含み損益の最大値
        self.provider.resetHoldPosCounter()

    def calc_penalty_trade_count(self) -> float:
        """
        取引回数に応じたペナルティ
        """
        penalty = self.provider.n_trade * self.penalty_trade_count
        return np.tanh(penalty)

    def evalReward(self, action: int) -> float:
        action_type = ActionType(action)
        reward = 0.0
        if self.position == PositionType.NONE:
            # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
            # ポジションが無い場合に取りうるアクションは HOLD, BUY, SELL
            # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
            if action_type == ActionType.HOLD:
                # HOLD カウンターのインクリメント
                self.provider.n_hold += 1.0
                reward += self.reward_penalty_hold
            elif action_type == ActionType.BUY:
                # HOLD カウンターのリセット
                self.provider.n_hold = 0.0
                # =============================================================
                # 買建 (LONG)
                # =============================================================
                # 取引コストペナルティ付与
                reward -= self.calc_penalty_trade_count()
                self.provider.n_trade += 1  # 取引回数を更新
                self.position = PositionType.LONG  # ポジションを更新
                self.price_entry = self.provider.price  # 取得価格
                reward += np.tanh(
                    (self.provider.vwap - self.price_entry) / self.provider.vwap * self.factor_vwap_scaling)
                # -------------------------------------------------------------
                # 取引明細
                # -------------------------------------------------------------
                self.add_transaction("買建")
            elif action_type == ActionType.SELL:
                # HOLD カウンターのリセット
                self.provider.n_hold = 0.0
                # =============================================================
                # 売建 (SHORT)
                # =============================================================
                # 取引コストペナルティ付与
                reward -= self.calc_penalty_trade_count()
                self.provider.n_trade += 1  # 取引回数を更新
                self.position = PositionType.SHORT  # ポジションを更新
                self.price_entry = self.provider.price  # 取得価格
                reward += np.tanh(
                    (self.price_entry - self.provider.vwap) / self.provider.vwap * self.factor_vwap_scaling)
                # -------------------------------------------------------------
                # 取引明細
                # -------------------------------------------------------------
                self.add_transaction("売建")
            else:
                raise TypeError(f"Unknown ActionType: {action_type}")
        elif self.position == PositionType.LONG:
            # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
            # LONG ポジションの場合に取りうるアクションは HOLD, SELL
            # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
            if action_type == ActionType.HOLD:
                # =============================================================
                # 含み益
                # =============================================================
                profit = self.get_profit()
                if self.profit_max < profit:
                    self.profit_max = profit
                # 含み益を持ち続けることで付与されるボーナス
                self.provider.n_hold_position += 1
                k = self.provider.n_hold_position * self.ratio_hold_position
                profit_weighted = profit * (1 + k)
                reward += self.get_reward_sqrt(profit_weighted) * self.ratio_unreal_profit
            elif action_type == ActionType.BUY:
                # 取引ルール違反
                raise TypeError(f"Violation of transaction rule: {action_type}")
            elif action_type == ActionType.SELL:
                # =============================================================
                # 売埋
                # =============================================================
                # 取引コストペナルティ付与
                reward -= self.calc_penalty_trade_count()
                self.provider.n_trade += 1  # 取引回数を更新
                # 含み損益 →　確定損益
                profit = self.get_profit()
                # 損益追加
                self.pnl_total += profit
                # 報酬
                reward += self.get_reward_sqrt(profit)
                # -------------------------------------------------------------
                # 取引明細
                # -------------------------------------------------------------
                # 返済: 買建 (LONG) → 売埋
                self.add_transaction("売埋", profit)
                # =============================================================
                # ポジション解消
                # =============================================================
                self.clear_position()
            else:
                raise TypeError(f"Unknown ActionType: {action_type}")
        elif self.position == PositionType.SHORT:
            # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
            # SHORT ポジションの場合に取りうるアクションは HOLD, BUY
            # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
            if action_type == ActionType.HOLD:
                # =============================================================
                # 含み益
                # =============================================================
                profit = self.get_profit()
                if self.profit_max < profit:
                    self.profit_max = profit
                # 含み益を持ち続けることで付与されるボーナス
                self.provider.n_hold_position += 1
                k = self.provider.n_hold_position * self.ratio_hold_position
                profit_weighted = profit * (1 + k)
                reward += self.get_reward_sqrt(profit_weighted) * self.ratio_unreal_profit
            elif action_type == ActionType.BUY:
                # =============================================================
                # 買埋
                # =============================================================
                # 取引コストペナルティ付与
                reward -= self.calc_penalty_trade_count()
                self.provider.n_trade += 1  # 取引回数を更新
                # 含み損益 →　確定損益
                profit = self.get_profit()
                # 損益追加
                self.pnl_total += profit
                # 報酬
                reward += self.get_reward_sqrt(profit)
                # -------------------------------------------------------------
                # 取引明細
                # -------------------------------------------------------------
                # 返済: 売建 (SHORT) → 買埋
                self.add_transaction("買埋", profit)
                # =============================================================
                # ポジション解消
                # =============================================================
                self.clear_position()
            elif action_type == ActionType.SELL:
                # 取引ルール違反
                raise TypeError(f"Violation of transaction rule: {action_type}")
            else:
                raise TypeError(f"Unknown ActionType: {action_type}")
        else:
            raise TypeError(f"Unknown PositionType: {self.position}")

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
        reward += self.get_reward_sqrt(profit)
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

    def get_reward_sqrt(self, profit: float) -> float:
        # 報酬は呼び値で割る
        return np.sign(profit) * np.sqrt(abs(profit / self.tickprice)) / self.factor_reward_sqrt

    def getNumberOfTransactions(self) -> int:
        return len(self.dict_transaction["注文日時"])

    def getPL4Obs(self) -> float:
        """
        観測値用に、損益用の報酬と同じにスケーリングして含み損益を返す。
        """
        profit = self.get_profit()
        return self.get_reward_sqrt(profit)

    def getPLRaw(self) -> float:
        """
        含み損益。
        """
        return self.get_profit()

    def getPLMax4Obs(self) -> float:
        """
        含み損益最大値からの比。
        """
        if self.profit_max == 0:
            return 0.0
        else:
            return self.get_reward_sqrt(self.profit_max)

    def getPLMaxRaw(self) -> float:
        """
        含み損益最大値。
        """
        return self.profit_max

    def getPLRatio4Obs(self) -> float:
        """
        含み損益最大値からの比。
        """
        if self.profit_max == 0:
            return 0.0
        else:
            return self.get_profit() / self.profit_max

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
