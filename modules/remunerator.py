import datetime

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

    def add_transaction(self, transaction: str, profit: float = np.nan):
        self.provider.dict_transaction["注文日時"].append(self.get_datetime(self.provider.ts))
        self.provider.dict_transaction["銘柄コード"].append(self.provider.code)
        self.provider.dict_transaction["売買"].append(transaction)
        self.provider.dict_transaction["約定単価"].append(self.provider.price)
        self.provider.dict_transaction["約定数量"].append(self.provider.unit)
        self.provider.dict_transaction["損益"].append(profit)

    def clear(self):
        pass
        """
        self.clear_position()
        self.pnl_total = 0.0  # 総損益
        self.provider.resetTradeCounter()  # 取引回数カウンターのリセット
        self.dict_transaction = self.init_transaction()  # 取引明細
        # ポジション無し HOLD カウンタのリセット
        self.provider.n_hold = 0
        """

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
        profit = self.get_profit()
        if self.provider.position == PositionType.LONG:
            # 返済: 買建 (LONG) → 売埋
            # -------------------------------------------------------------
            # 取引明細
            # -------------------------------------------------------------
            self.add_transaction("売埋（強制返済）", profit)
        elif self.provider.position == PositionType.SHORT:
            # 返済: 売建 (SHORT) → 買埋
            # -------------------------------------------------------------
            # 取引明細
            # -------------------------------------------------------------
            self.add_transaction("買埋（強制返済）", profit)
        else:
            # ポジション無し
            pass
        # 損益追加
        self.provider.pnl_total += profit
        # 報酬
        reward += self.get_profit_scaled(profit)  # シンプルにスケーリングされた報酬

        return reward

    @staticmethod
    def get_datetime(t: float) -> str:
        return str(datetime.datetime.fromtimestamp(int(t)))

    def get_profit(self) -> float:
        if self.provider.position == PositionType.LONG:
            # 返済: 買建 (LONG) → 売埋
            profit = self.provider.price - self.provider.price_entry
        elif self.provider.position == PositionType.SHORT:
            # 返済: 売建 (SHORT) → 買埋
            profit = self.provider.price_entry - self.provider.price
        else:
            profit = 0.0  # 実現損益

        # 最大含み益を保持
        if self.provider.profit_max < profit:
            self.provider.profit_max = profit

        return profit

    def get_profit_scaled(self, profit) -> float:
        return np.tanh(profit / self.provider.price_tick / self.DIVISOR_PROFIT_SCALED)

    def getNumberOfTransactions(self) -> int:
        return len(self.provider.dict_transaction["注文日時"])

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
        return self.get_profit_scaled(self.provider.profit_max)  # 報酬用にスケーリング

    def getPLMaxRaw(self) -> float:
        """
        含み損益最大値。
        """
        return self.provider.profit_max

    def getPLRatio4Obs(self) -> float:
        """
        含み損益最大値からの比。
        """
        if self.provider.profit_max == 0:
            return 0.0
        else:
            return self.get_profit() / self.provider.profit_max


    def position_close(self) -> float:
        reward = 0

        # HOLD カウンター（建玉あり）のリセット
        self.provider.n_hold_position = 0
        # 取引回数のインクリメント
        self.provider.n_trade += 1

        # 確定損益
        profit = self.get_profit()
        # 確定損益追加
        self.provider.pnl_total += profit
        # 報酬に追加
        reward += self.get_profit_scaled(profit)

        # エントリ価格をリセット
        self.provider.price_entry = 0.0
        # 含み損益の最大値
        self.provider.profit_max = 0.0

        # -------------------------------------------------------------
        # 取引明細
        # -------------------------------------------------------------
        if self.provider.position == PositionType.LONG:
            # 返済: 買建 (LONG) → 売埋
            self.add_transaction("売埋", profit)
        elif self.provider.position == PositionType.SHORT:
            # 返済: 売建 (SHORT) → 買埋
            self.add_transaction("買埋", profit)
        else:
            raise TypeError(f"Unknown PositionType: {self.provider.position}")

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

        # MAΔS に従った場合
        signal = SignalSign(self.provider.mad_sign_signal)
        if signal == SignalSign.POSITIVE:
            # ゴールデンクロスは買いシグナル
            if position == PositionType.LONG:
                reward += self.REWARD_POSITION  # テクニカル指標に従ったポジション
            elif position == PositionType.SHORT:
                reward -= self.PENALTY_POSITION  # テクニカル指標に逆行したポジション
            else:
                raise TypeError(f"Unknown PositionType: {self.provider.position}")
        elif signal == SignalSign.NEGATIVE:
            # デッドクロスは売りシグナル
            if position == PositionType.LONG:
                reward -= self.PENALTY_POSITION  # テクニカル指標に逆行したポジション
            elif position == PositionType.SHORT:
                reward += self.REWARD_POSITION  # テクニカル指標に従ったポジション
            else:
                raise TypeError(f"Unknown PositionType: {self.provider.position}")
        else:
            pass

        # -------------------------------------------------------------
        # 取引明細
        # -------------------------------------------------------------
        if self.provider.position == PositionType.LONG:
            self.add_transaction("買建")
        elif self.provider.position == PositionType.SHORT:
            self.add_transaction("売建")
        else:
            raise TypeError(f"Unknown PositionType: {self.provider.position}")

        return reward
