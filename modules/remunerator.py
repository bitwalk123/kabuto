import datetime

import numpy as np

from modules.feature_provider import FeatureProvider
from structs.app_enum import PositionType, ActionType


class RewardManager:
    """
    アクションに対して報酬の支払うクラス
    方策マスクでナンピンをしないことが前提
    """

    def __init__(self, provider: FeatureProvider, code: str):
        # 特徴量プロバイダ
        self.provider = provider
        self.code: str = code  # 銘柄コード

        # ---------------------------------------------------------------------
        # 取引関連
        # ---------------------------------------------------------------------
        self.dict_transaction = self.init_transaction()  # 取引明細
        self.position = PositionType.NONE  # ポジション（建玉）
        self.pnl_total = 0.0  # 総損益
        self.price_tick: float = 1.0  # 呼び値
        self.price_entry = 0.0  # 取得価格
        self.profit_max = 0.0  # 含み損益の最大値
        self.unit: float = 1.0  # 売買単位
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # 報酬設計
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        self.divisor_profit_scaled = 100.0  # 損益を報酬化する際の除数

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
        # ポジション無し HOLD カウンタのリセット
        self.provider.n_hold = 0

    def clear_position(self):
        self.position = PositionType.NONE
        # エントリ価格をリセット
        self.price_entry = 0.0
        # 含み損益の最大値
        self.profit_max = 0.0
        # ポジション持ち HOLD カウンタのリセット
        self.provider.n_hold_position = 0

    def evalReward(self, action: int) -> float:
        action_type = ActionType(action)
        reward = 0.0
        if self.position == PositionType.NONE:
            # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
            # ポジションが無い場合に取りうるアクションは HOLD, BUY, SELL
            # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
            if action_type == ActionType.HOLD:
                # HOLD カウンター（建玉なし）のインクリメント
                self.provider.n_hold += 1
            elif action_type == ActionType.BUY:
                # =============================================================
                # 買建 (LONG)
                # =============================================================
                # 新規ポジション
                reward += self.position_open(PositionType.LONG)
            elif action_type == ActionType.SELL:
                # =============================================================
                # 売建 (SHORT)
                # =============================================================
                # 新規ポジション
                reward += self.position_open(PositionType.SHORT)
            else:
                raise TypeError(f"Unknown ActionType: {action_type}")
        elif self.position == PositionType.LONG:
            # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
            # LONG ポジションの場合に取りうるアクションは HOLD, SELL
            # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
            if action_type == ActionType.HOLD:
                # HOLD カウンター（建玉あり）のインクリメント
                self.provider.n_hold_position += 1
            elif action_type == ActionType.BUY:
                # 取引ルール違反
                raise TypeError(f"Violation of transaction rule: {action_type}")
            elif action_type == ActionType.SELL:
                # =============================================================
                # 売埋
                # =============================================================
                reward += self.position_close()
            else:
                raise TypeError(f"Unknown ActionType: {action_type}")
        elif self.position == PositionType.SHORT:
            # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
            # SHORT ポジションの場合に取りうるアクションは HOLD, BUY
            # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
            if action_type == ActionType.HOLD:
                # HOLD カウンター（建玉あり）のインクリメント
                self.provider.n_hold_position += 1
            elif action_type == ActionType.BUY:
                # =============================================================
                # 買埋
                # =============================================================
                reward += self.position_close()
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
        reward += self.get_profit_scaled(profit)  # シンプルにスケーリングされた報酬

        return reward

    @staticmethod
    def get_datetime(t: float) -> str:
        return str(datetime.datetime.fromtimestamp(int(t)))

    def get_profit(self) -> float:
        if self.position == PositionType.LONG:
            # 返済: 買建 (LONG) → 売埋
            profit = self.provider.price - self.price_entry
        elif self.position == PositionType.SHORT:
            # 返済: 売建 (SHORT) → 買埋
            profit = self.price_entry - self.provider.price
        else:
            profit = 0.0  # 実現損益

        # 最大含み益を保持
        if self.profit_max < profit:
            self.profit_max = profit

        return profit

    def get_profit_scaled(self, profit) -> float:
        return np.tanh(profit / self.price_tick / self.divisor_profit_scaled)

    def getNumberOfTransactions(self) -> int:
        return len(self.dict_transaction["注文日時"])

    def getPL4Obs(self) -> float:
        """
        観測値用に、損益用の報酬と同じにスケーリングして含み損益を返す。
        """
        profit = self.get_profit()
        return self.get_profit_scaled(profit)  # 報酬用にスケーリング

    def getPLRaw(self) -> float:
        """
        含み損益。
        """
        return self.get_profit()

    def getPLMax4Obs(self) -> float:
        """
        含み損益最大値
        """
        return self.get_profit_scaled(self.profit_max)  # 報酬用にスケーリング

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

    def position_close(self) -> float:
        reward = 0

        # HOLD カウンター（建玉あり）のリセット
        self.provider.n_hold_position = 0
        # 取引回数のインクリメント
        self.provider.n_trade += 1

        # 確定損益
        profit = self.get_profit()
        # 確定損益追加
        self.pnl_total += profit
        # 報酬に追加
        reward += self.get_profit_scaled(profit)

        # エントリ価格をリセット
        self.price_entry = 0.0
        # 含み損益の最大値
        self.profit_max = 0.0

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
        self.provider.n_hold = 0
        # 取引回数のインクリメント
        self.provider.n_trade += 1

        # エントリ価格
        self.price_entry = self.provider.price
        # ポジションを更新
        self.position = position

        # -------------------------------------------------------------
        # 取引明細
        # -------------------------------------------------------------
        if self.position == PositionType.LONG:
            self.add_transaction("買建")
        elif self.position == PositionType.SHORT:
            self.add_transaction("売建")
        else:
            raise TypeError(f"Unknown PositionType: {self.position}")

        return reward
