import numpy as np


class TradingEnv:
    """
    シンプルなデイトレード用強化学習環境
    - アクション: 0=HOLD, 1=BUY, 2=SELL, 3=REPAY
    - ナンピン禁止
    - 建玉は1つだけ (BUY or SELL)
    - REPAYでのみ損益確定
    """

    def __init__(self, prices, penalty=-1.0):
        self.prices = prices
        self.n_steps = len(prices)
        self.penalty = penalty

        # 状態変数
        self.reset()

    def reset(self):
        self.current_step = 0
        self.position = 0  # 0=ノーポジ, 1=BUY, -1=SELL
        self.entry_price = None
        self.total_profit = 0.0
        return self._get_state()

    def _get_state(self):
        """
        状態ベクトルを返す
        - 直近価格
        - ポジション状態
        """
        price = self.prices[self.current_step]
        return np.array([price, self.position], dtype=np.float32)

    def step(self, action):
        """
        環境の1ステップを実行
        action: 0=HOLD, 1=BUY, 2=SELL, 3=REPAY
        """
        done = False
        reward = 0.0
        info = {}

        price = self.prices[self.current_step]

        # --- ノーポジ時 ---
        if self.position == 0:
            if action == 1:  # BUY
                self.position = 1
                self.entry_price = price
            elif action == 2:  # SELL
                self.position = -1
                self.entry_price = price
            elif action == 3:  # REPAY → 無効
                reward = self.penalty
                info["reason"] = "invalid_repay"

        # --- BUY中 ---
        elif self.position == 1:
            if action == 3:  # REPAY
                profit = price - self.entry_price
                reward = profit
                self.total_profit += profit
                self.position = 0
                self.entry_price = None
                info["action"] = "repay"
            elif action in [1, 2]:  # ナンピン禁止
                reward = self.penalty
                info["reason"] = "forbidden_action"

        # --- SELL中 ---
        elif self.position == -1:
            if action == 3:  # REPAY
                profit = self.entry_price - price
                reward = profit
                self.total_profit += profit
                self.position = 0
                self.entry_price = None
                info["action"] = "repay"
            elif action in [1, 2]:  # ナンピン禁止
                reward = self.penalty
                info["reason"] = "forbidden_action"

        # --- 次のステップへ ---
        self.current_step += 1
        if self.current_step >= self.n_steps - 1:
            done = True
            # 最後に建玉を残したまま終わった場合は強制決済
            if self.position != 0:
                if self.position == 1:
                    profit = price - self.entry_price
                else:  # self.position == -1
                    profit = self.entry_price - price
                self.total_profit += profit
                reward += profit
                info["forced_repay"] = True
                self.position = 0
                self.entry_price = None

        return self._get_state(), reward, done, info
