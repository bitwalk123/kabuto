import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# -------------------------------
# 特徴量計算ユーティリティ
# -------------------------------
def compute_features(df, n=60):
    df = df.copy()
    df['DeltaVolume'] = df['Volume'].diff().fillna(0)
    df['LogDeltaVolume'] = np.log1p(df['DeltaVolume'])
    df['MA'] = df['Price'].rolling(n).mean()
    df['STD'] = df['Price'].rolling(n).std()
    delta = df['Price'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(n).mean()
    roll_down = down.rolling(n).mean()
    df['RSI'] = 100 - 100 / (1 + roll_up / roll_down)
    df['ZScore'] = (df['Price'] - df['MA']) / df['STD']
    df.fillna(0, inplace=True)
    return df[['LogDeltaVolume', 'MA', 'STD', 'RSI', 'ZScore']].values

# -------------------------------
# PPO ネットワーク
# -------------------------------
class PolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.fc(x)

class ValueNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.fc(x)

# -------------------------------
# Trading Simulator
# -------------------------------
class TradingSimulator:
    ACTIONS = {0: "HOLD", 1: "BUY", 2: "SELL", 3: "REPAY"}

    def __init__(self, model_path="policy.pth", slippage=1, unit=100, device="cpu"):
        self.slippage = slippage
        self.unit = unit
        self.device = device
        self.position = 0  # +1=ロング, -1=ショート, 0=なし
        self.entry_price = 0
        self.history = deque(maxlen=60)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model {model_path} not found.")
        self.policy = torch.load(model_path, map_location=device)
        self.policy.eval()

    def add(self, time, price, volume, force_close=False):
        self.history.append((price, volume))
        if len(self.history) < 60:
            return "HOLD"

        # 特徴量作成
        df = pd.DataFrame(self.history, columns=["Price", "Volume"])
        features = compute_features(df)[-1]
        state = torch.tensor(features, dtype=torch.float32).to(self.device)

        # 方策推論（ε-greedy で少しランダム性）
        with torch.no_grad():
            logits = self.policy(state)
            probs = torch.softmax(logits, dim=0).cpu().numpy()

        eps = 0.05
        if np.random.rand() < eps:
            action = np.random.choice(len(probs))
        else:
            action = int(np.argmax(probs))

        # 強制返済
        if force_close and self.position != 0:
            action = 3  # REPAY

        # 建玉管理
        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = price + self.slippage
        elif action == 2 and self.position == 0:
            self.position = -1
            self.entry_price = price - self.slippage
        elif action == 3 and self.position != 0:
            self.position = 0
            self.entry_price = 0

        return self.ACTIONS[action]

# -------------------------------
# PPO Trainer
# -------------------------------
class Trainer:
    def __init__(self, input_dim=5, output_dim=4, gamma=0.99, lr=3e-4, device="cpu", model_path="policy.pth"):
        self.device = device
        self.gamma = gamma
        self.model_path = model_path

        self.policy = PolicyNet(input_dim, output_dim).to(device)
        self.value = ValueNet(input_dim).to(device)
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optim = optim.Adam(self.value.parameters(), lr=lr)

    def train(self, df):
        df = df.copy()
        df['Profit'] = 0
        df['Action'] = "HOLD"

        position = 0
        entry_price = 0
        history = deque(maxlen=60)

        for i, row in df.iterrows():
            price = row['Price']
            volume = row['Volume']
            history.append((price, volume))

            if len(history) < 60:
                df.at[i, 'Action'] = "HOLD"
                continue

            features = compute_features(pd.DataFrame(history, columns=["Price","Volume"]))[-1]
            state = torch.tensor(features, dtype=torch.float32).to(self.device)
            logits = self.policy(state)
            probs = torch.softmax(logits, dim=0).detach().cpu().numpy()

            eps = 0.05
            if np.random.rand() < eps:
                action = np.random.choice(len(probs))
            else:
                action = int(np.argmax(probs))

            # 建玉管理と報酬計算
            reward = 0
            if action == 1 and position == 0:
                position = 1
                entry_price = price + 1
            elif action == 2 and position == 0:
                position = -1
                entry_price = price - 1
            elif action == 3 and position != 0:
                reward = (price - entry_price) * position * 100 if position == 1 else (entry_price - price) * 100
                df.at[i, 'Profit'] = reward
                position = 0
                entry_price = 0

            # 保有中の報酬（含み益の5%を毎ティック加算）
            if position != 0:
                unrealized = (price - entry_price) * position * 100 if position == 1 else (entry_price - price) * 100
                reward += 0.05 * unrealized

            df.at[i, 'Action'] = TradingSimulator.ACTIONS[action]

            # --- PPO 更新（簡易版） ---
            self.update_model(state, action, reward)

        # 最終行で強制返済
        if position != 0:
            price = df.iloc[-1]['Price']
            reward = (price - entry_price) * position * 100 if position == 1 else (entry_price - price) * 100
            df.at[df.index[-1], 'Profit'] += reward
            df.at[df.index[-1], 'Action'] = "REPAY"

        self.save_model()
        return df

    def update_model(self, state, action, reward):
        # 状態価値
        value = self.value(state)
        advantage = reward - value.item()

        # Policy 更新
        logits = self.policy(state)
        log_probs = torch.log_softmax(logits, dim=0)
        loss_policy = -log_probs[action] * advantage
        self.policy_optim.zero_grad()
        loss_policy.backward()
        self.policy_optim.step()

        # Value 更新
        loss_value = (value - reward) ** 2
        self.value_optim.zero_grad()
        loss_value.backward()
        self.value_optim.step()

    def save_model(self):
        torch.save(self.policy, self.model_path)

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"{self.model_path} not found.")
        self.policy = torch.load(self.model_path, map_location=self.device)
