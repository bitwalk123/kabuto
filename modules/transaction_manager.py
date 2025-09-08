from modules.trading_env_20250908_1 import ActionType, PositionType


class TransactionManager:
    def __init__(self):
        self.reward_none = 0.0  # 報酬なし
        self.reward_rule = 1.0  # ルール適合報酬
        self.reward_rule_dbl = 2.0  # ルール適合報酬（大きめ）
        self.penalty_rule = -10.0  # ルール違反ペナルティ
        self.penalty_rule_dbl = -20.0  # ルール違反ペナルティ（厳しめ）

        self.reward_pnl_ratio = 0.1  # 含み損益に対する報酬比
        self.penalty_bit = -0.01  #

        self.position = PositionType.NONE
        self.price_entry = 0.0
        self.action_pre = ActionType.HOLD
        self.pnl_total = 0

    def clearPosition(self):
        self.position = PositionType.NONE
        self.price_entry = 0.0

    def clearAll(self):
        self.clearPosition()
        self.action_pre = ActionType.HOLD
        self.pnl_total = 0

    def has_position(self) -> bool:
        if self.price_entry > 0:
            return True
        else:
            return False

    def setAction(self, action: ActionType, price: float) -> float:
        reward = 0
        # 売買ルール
        if self.has_position():  # 建玉あり
            if self.action_pre == ActionType.HOLD:
                if action == ActionType.HOLD:
                    reward += self.reward_none
                elif action == ActionType.BUY:
                    reward += self.penalty_rule
                elif action == ActionType.SELL:
                    reward += self.penalty_rule
                elif action == ActionType.REPAY:
                    reward += self.reward_rule_dbl
                else:
                    pass
            elif self.action_pre == ActionType.BUY:
                if action == ActionType.HOLD:
                    reward += self.reward_none
                elif action == ActionType.BUY:
                    reward += self.penalty_rule_dbl
                elif action == ActionType.SELL:
                    reward += self.penalty_rule
                elif action == ActionType.REPAY:
                    reward += self.reward_rule_dbl
                else:
                    pass
            elif self.action_pre == ActionType.SELL:
                if action == ActionType.HOLD:
                    reward += self.reward_none
                elif action == ActionType.BUY:
                    reward += self.penalty_rule
                elif action == ActionType.SELL:
                    reward += self.penalty_rule_dbl
                elif action == ActionType.REPAY:
                    reward += self.reward_rule_dbl
                else:
                    pass
            elif self.action_pre == ActionType.REPAY:
                if action == ActionType.HOLD:
                    reward += self.reward_none
                elif action == ActionType.BUY:
                    reward += self.penalty_rule
                elif action == ActionType.SELL:
                    reward += self.penalty_rule
                elif action == ActionType.REPAY:
                    reward += self.penalty_rule_dbl
                else:
                    pass
        else:  # 建玉なし
            if self.action_pre == ActionType.HOLD:
                if action == ActionType.HOLD:
                    reward += self.penalty_bit
                elif action == ActionType.BUY:
                    reward += self.reward_rule_dbl
                elif action == ActionType.SELL:
                    reward += self.reward_rule_dbl
                elif action == ActionType.REPAY:
                    reward += self.penalty_rule
                else:
                    pass
            elif self.action_pre == ActionType.BUY:
                if action == ActionType.HOLD:
                    reward += self.reward_none
                elif action == ActionType.BUY:
                    reward += self.reward_rule
                elif action == ActionType.SELL:
                    reward += self.reward_rule
                elif action == ActionType.REPAY:
                    reward += self.penalty_rule
                else:
                    pass
            elif self.action_pre == ActionType.SELL:
                if action == ActionType.HOLD:
                    reward += self.reward_none
                elif action == ActionType.BUY:
                    reward += self.reward_rule
                elif action == ActionType.SELL:
                    reward += self.reward_rule
                elif action == ActionType.REPAY:
                    reward += self.penalty_rule
                else:
                    pass
            elif self.action_pre == ActionType.REPAY:
                if action == ActionType.HOLD:
                    reward += self.reward_none
                elif action == ActionType.BUY:
                    reward += self.reward_rule
                elif action == ActionType.SELL:
                    reward += self.reward_rule
                elif action == ActionType.REPAY:
                    reward += self.penalty_rule_dbl
                else:
                    pass

        # 一つ前のアクションを更新
        self.action_pre = action

        # 建玉損益
        if self.position == PositionType.LONG:
            pnl = price - self.price_entry
            if action == ActionType.REPAY:  # 利確
                reward += pnl
                self.pnl_total += pnl
                self.clearPosition()
            else:  # 含み損益
                reward += pnl * self.reward_pnl_ratio
        elif self.position == PositionType.SHORT:
            pnl = self.price_entry - price
            if action == ActionType.REPAY:  # 利確
                reward += pnl
                self.pnl_total += pnl
                self.clearPosition()
            else:  # 含み損益
                reward += pnl * self.reward_pnl_ratio
        elif self.position == PositionType.NONE:
            if action == ActionType.BUY:  # 買建
                self.position = PositionType.LONG
                self.price_entry = price
            elif action == ActionType.SELL:  # 売建（空売り）
                self.position = PositionType.SHORT
                self.price_entry = price
            else:
                pass
        else:
            pass

        return reward
