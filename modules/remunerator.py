import numpy as np

from modules.feature_provider import FeatureProvider
from structs.app_enum import PositionType, ActionType, SignalSign


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

    def clear_position(self):
        self.provider.position = PositionType.NONE
        # エントリ価格をリセット
        self.provider.price_entry = 0.0
        # 含み損益の最大値
        self.provider.profit_max = 0.0
        # ポジション持ち HOLD カウンタのリセット
        self.provider.n_hold_position = 0

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
                reward += self.position_open(PositionType.LONG)
            elif action_type == ActionType.SELL:
                # =============================================================
                # 売建 (SHORT)
                # =============================================================
                # 新規ポジション
                reward += self.position_open(PositionType.SHORT)
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
                reward += self.position_close()
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
                reward += self.position_close()
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

    def position_close(self) -> float:
        reward = 0

        # HOLD カウンター（建玉あり）のリセット
        self.provider.n_hold_position = 0
        # 取引回数のインクリメント
        self.provider.n_trade += 1

        # 確定損益
        profit = self.provider.get_profit()
        # 確定損益追加
        self.provider.pnl_total += profit
        # 報酬に追加
        reward += self.get_profit_scaled(profit)

        # エントリ価格をリセット
        self.provider.price_entry = 0.0
        # 含み損益の最大値
        self.provider.profit_max = 0.0

        # 取引明細更新（建玉返済）
        self.provider.transaction_close(profit)

        # ポジションの更新
        self.provider.position = PositionType.NONE

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
        self.provider.price_entry = self.provider.price
        # ポジションを更新
        self.provider.position = position
        # 取引明細更新（新規建玉）
        self.provider.transaction_open()

        return reward
