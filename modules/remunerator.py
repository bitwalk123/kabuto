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
        # 銘柄コード
        self.provider.code = code
        # ---------------------------------------------------------------------
        # 取引関連
        # ---------------------------------------------------------------------
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # 報酬設計
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        self.DIVISOR_PROFIT_SCALED = 100.0  # 損益を報酬化する際の除数
        self.REWARD_POSITION = 0.1  # テクニカル指標に従ったポジション
        self.PENALTY_POSITION = 0.1  # テクニカル指標に逆行したポジション

    def evalReward(self, action: int) -> float:
        action_type = ActionType(action)
        reward = 0.0
        if self.provider.position == PositionType.NONE:
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
                reward += self.provider.position_open(PositionType.LONG)
            elif action_type == ActionType.SELL:
                # =============================================================
                # 売建 (SHORT)
                # =============================================================
                # 新規ポジション
                reward += self.provider.position_open(PositionType.SHORT)
            else:
                raise TypeError(f"Unknown ActionType: {action_type}")
        elif self.provider.position == PositionType.LONG:
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
                reward += self.get_profit_scaled(self.provider.position_close())
            else:
                raise TypeError(f"Unknown ActionType: {action_type}")
        elif self.provider.position == PositionType.SHORT:
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
                reward += self.get_profit_scaled(self.provider.position_close())
            elif action_type == ActionType.SELL:
                # 取引ルール違反
                raise TypeError(f"Violation of transaction rule: {action_type}")
            else:
                raise TypeError(f"Unknown ActionType: {action_type}")
        else:
            raise TypeError(f"Unknown PositionType: {self.provider.position}")

        return reward

    def forceRepay(self) -> float:
        reward = 0.0
        profit = self.provider.get_profit()
        if self.provider.position == PositionType.LONG:
            # 返済: 買建 (LONG) → 売埋
            # -------------------------------------------------------------
            # 取引明細
            # -------------------------------------------------------------
            self.provider.add_transaction("売埋（強制返済）", profit)
        elif self.provider.position == PositionType.SHORT:
            # 返済: 売建 (SHORT) → 買埋
            # -------------------------------------------------------------
            # 取引明細
            # -------------------------------------------------------------
            self.provider.add_transaction("買埋（強制返済）", profit)
        else:
            # ポジション無し
            pass
        # 損益追加
        self.provider.pnl_total += profit
        # 報酬
        reward += self.get_profit_scaled(profit)  # シンプルにスケーリングされた報酬

        return reward

    def get_profit_scaled(self, profit) -> float:
        return np.tanh(profit / self.provider.price_tick / self.DIVISOR_PROFIT_SCALED)

    def getNumberOfTransactions(self) -> int:
        return len(self.provider.dict_transaction["注文日時"])

