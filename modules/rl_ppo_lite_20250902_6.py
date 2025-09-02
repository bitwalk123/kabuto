import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------
# ユーティリティ関数
# ---------------------------

def compute_features(df):
    df = df.copy()
    df['DeltaVolume'] = df['Volume'].diff().fillna(0)
    df['LogDeltaVolume'] = np.log1p(df['DeltaVolume'])
    df['MA'] = df['Price'].rolling(60).mean()
    df['STD'] = df['Price'].rolling(60).std()
    df['RSI'] = compute_rsi(df['Price'], n=60)
    df['ZScore'] = (df['Price'] - df['MA']) / df['STD']
    df.fillna(0, inplace=True)
    return df[['LogDeltaVolume', 'MA', 'STD', 'RSI', 'ZScore']].values

def compute_rsi(prices, n=60):
    delta = prices.diff()
    up = np.where(delta > 0, delta, 0)
    down = np.where(delta < 0, -delta, 0)
    roll_up = pd.Series(up).rolling(n).mean()
    roll_down = pd.Series(down).rolling(n).mean()
    rs = roll_up / (roll_down + 1e-6)
    rsi = 100 - 100 / (1 + rs)
    return rsi

# ---------------------------
# PPO モデル
# ---------------------------

class PolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        return self.net(x)

class ValueNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

# ---------------------------
# Trading Simulator
# ---------------------------

class TradingSimulator:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_path} が存在しません")
        self.device = torch.device("cpu")
        self.policy = torch.load(model_path, map_location=self.device)
        self.position = 0   # 1:ロング, -1:ショート, 0:なし
        self.entry_price = 0
        self.slippage = 1
        self.unit = 100

    def add(self, time: float, price: float, volume: float) -> str:
        # 特徴量を簡易生成
        feature = np.array([price, volume], dtype=np.float32)
        feature = torch.tensor(feature).float().unsqueeze(0)
        probs = self.policy(feature).detach().numpy()[0]
        action_idx = np.argmax(probs)
        action_map = {0: "HOLD", 1: "BUY", 2: "SELL", 3: "REPAY"}
        action = action_map[action_idx]

        # 制約条件
        if self.position != 0:
            if action in ["BUY", "SELL"]:
                action = "HOLD"

        # ポジション更新
        if action == "BUY":
            self.position = 1
            self.entry_price = price + self.slippage
        elif action == "SELL":
            self.position = -1
            self.entry_price = price - self.slippage
        elif action == "REPAY":
            self.position = 0
            self.entry_price = 0

        return action

# ---------------------------
# Trainer
# ---------------------------

class Trainer:
    def __init__(self):
        self.device = torch.device("cpu")
        self.input_dim = 5  # LogDeltaVolume, MA, STD, RSI, ZScore
        self.policy_net = PolicyNet(self.input_dim).to(self.device)
        self.value_net = ValueNet(self.input_dim).to(self.device)
        self.optimizer_policy = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.optimizer_value = optim.Adam(self.value_net.parameters(), lr=1e-3)
        self.gamma = 0.99
        self.eps = 0.2  # PPOクリッピング

    def train(self, df: pd.DataFrame) -> pd.DataFrame:
        features = compute_features(df)
        df_transaction = pd.DataFrame(columns=["Time", "Price", "Action", "Profit"])
        position = 0
        entry_price = 0
        slippage = 1
        unit = 100

        for t in range(len(df)):
            if t < 60:
                action = "HOLD"
            else:
                state = torch.tensor(features[t]).float().unsqueeze(0)
                with torch.no_grad():
                    probs = self.policy_net(state)
                action_idx = np.random.choice(4, p=probs.numpy()[0])
                action_map = {0: "HOLD", 1: "BUY", 2: "SELL", 3: "REPAY"}
                action = action_map[action_idx]

                # 制約条件
                if position != 0 and action in ["BUY", "SELL"]:
                    action = "HOLD"

            # ポジション更新と利益計算
            profit = 0
            price = df.iloc[t]["Price"]
            if action == "BUY":
                position = 1
                entry_price = price + slippage
            elif action == "SELL":
                position = -1
                entry_price = price - slippage
            elif action == "REPAY":
                if position == 1:
                    profit = (price - slippage - entry_price) * unit
                elif position == -1:
                    profit = (entry_price - (price + slippage)) * unit
                position = 0
                entry_price = 0
            elif position != 0:
                # 保有中は含み損益の一部を報酬として加算
                if position == 1:
                    profit = (price - entry_price) * 0.1 * unit
                else:
                    profit = (entry_price - price) * 0.1 * unit

            df_transaction.loc[t] = [df.iloc[t]["Time"], price, action, profit]

        # 最終ティックで強制返済
        if position != 0:
            price = df.iloc[-1]["Price"]
            if position == 1:
                profit = (price - slippage - entry_price) * unit
            else:
                profit = (entry_price - (price + slippage)) * unit
            df_transaction.loc[len(df_transaction)] = [df.iloc[-1]["Time"], price, "REPAY", profit]

        # モデル保存
        torch.save(self.policy_net, "policy.pth")
        torch.save(self.value_net, "value.pth")
        return df_transaction

# ---------------------------
# 動作例
# ---------------------------

if __name__ == "__main__":
    # ダミーデータ
    np.random.seed(0)
    times = np.arange(19500)
    prices = np.cumsum(np.random.randn(19500)) + 1000
    volumes = np.cumsum(np.random.randint(1, 10, 19500))
    df = pd.DataFrame({"Time": times, "Price": prices, "Volume": volumes})

    trainer = Trainer()
    df_transaction = trainer.train(df)
    print(df_transaction.head())
    print("総利益:", df_transaction["Profit"].sum())
