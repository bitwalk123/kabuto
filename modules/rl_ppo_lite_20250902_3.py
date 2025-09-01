import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import gymnasium as gym

# ---------------------------
# ユーティリティ：特徴量生成
# ---------------------------
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["DeltaVolume"] = df["Volume"].diff().fillna(0)
    df["LogDeltaVolume"] = np.log1p(np.maximum(df["DeltaVolume"], 0))

    window = 60
    df["MA"] = df["Price"].rolling(window).mean()
    df["STD"] = df["Price"].rolling(window).std()
    df["RSI"] = compute_rsi(df["Price"], window)
    df["ZScore"] = (df["Price"] - df["MA"]) / df["STD"]

    df = df.fillna(0.0)
    return df

def compute_rsi(prices, window=60):
    delta = prices.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(window).mean()
    ma_down = down.rolling(window).mean()
    rsi = 100 * ma_up / (ma_up + ma_down + 1e-9)
    return rsi.fillna(50.0)

# ---------------------------
# PPO ネットワーク定義
# ---------------------------
class PolicyNet(nn.Module):
    def __init__(self, input_dim, action_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)

class ValueNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

# ---------------------------
# 簡易環境クラス
# ---------------------------
class TradingEnv(gym.Env):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = compute_features(df)
        self.action_space = gym.spaces.Discrete(4)  # HOLD, BUY, SELL, REPAY
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )
        self.reset()

    def reset(self, seed=None, options=None):
        self.t = 0
        self.position = None
        self.entry_price = 0
        self.done = False
        self.total_profit = 0
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.df.iloc[self.t]
        return np.array([
            row["LogDeltaVolume"],
            row["MA"], row["STD"], row["RSI"], row["ZScore"]
        ], dtype=np.float32)

    def step(self, action):
        price = self.df.iloc[self.t]["Price"]
        reward = 0.0

        # 制約：最初の60ティックは必ずHOLD
        if self.t < 60:
            action = 0  # HOLD

        # スリッページ処理
        slip = 1
        if action == 1 and self.position is None:  # BUY
            self.position = "LONG"
            self.entry_price = price + slip
        elif action == 2 and self.position is None:  # SELL
            self.position = "SHORT"
            self.entry_price = price - slip
        elif action == 3 and self.position is not None:  # REPAY
            if self.position == "LONG":
                exit_price = price - slip
                profit = (exit_price - self.entry_price) * 100
            else:
                exit_price = price + slip
                profit = (self.entry_price - exit_price) * 100
            reward += profit
            self.total_profit += profit
            self.position = None

        # 含み益の一部を報酬に加算
        if self.position is not None:
            if self.position == "LONG":
                unreal = (price - self.entry_price) * 100
            else:
                unreal = (self.entry_price - price) * 100
            reward += unreal * 0.001

        self.t += 1
        if self.t >= len(self.df):
            self.done = True
            # 強制返済
            if self.position is not None:
                if self.position == "LONG":
                    exit_price = price - slip
                    profit = (exit_price - self.entry_price) * 100
                else:
                    exit_price = price + slip
                    profit = (self.entry_price - exit_price) * 100
                reward += profit
                self.total_profit += profit
                self.position = None

        return self._get_obs(), reward, self.done, False, {}

# ---------------------------
# PPO トレーナー
# ---------------------------
class Trainer:
    def __init__(self, model_path="policy.pth"):
        self.model_path = model_path
        self.policy_net = None
        self.value_net = None

    def train(self, df: pd.DataFrame) -> pd.DataFrame:
        env = TradingEnv(df)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n

        self.policy_net = PolicyNet(obs_dim, act_dim)
        self.value_net = ValueNet(obs_dim)

        optimizer_p = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        optimizer_v = optim.Adam(self.value_net.parameters(), lr=1e-3)

        # 簡易版 PPO：1エポックのみ
        obs, _ = env.reset()
        done = False
        records = []

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            probs = self.policy_net(obs_tensor)
            action = torch.multinomial(probs, 1).item()

            next_obs, reward, done, _, _ = env.step(action)

            # advantage (簡易)
            value = self.value_net(obs_tensor)
            target = torch.tensor([[reward]], dtype=torch.float32)
            loss_v = (value - target).pow(2).mean()
            optimizer_v.zero_grad()
            loss_v.backward()
            optimizer_v.step()

            log_prob = torch.log(probs[0, action] + 1e-9)
            advantage = (target - value).detach()
            loss_p = -(log_prob * advantage)
            optimizer_p.zero_grad()
            loss_p.backward()
            optimizer_p.step()

            obs = next_obs

            action_str = ["HOLD", "BUY", "SELL", "REPAY"][action]
            records.append({
                "Time": df.iloc[env.t]["Time"] if env.t < len(df) else df.iloc[-1]["Time"],
                "Price": df.iloc[env.t-1]["Price"],
                "Action": action_str,
                "Profit": env.total_profit
            })

        torch.save(self.policy_net.state_dict(), self.model_path)
        return pd.DataFrame(records)

# ---------------------------
# 推論専用クラス
# ---------------------------
class TradingSimulator:
    def __init__(self, model_path="policy.pth"):
        if not os.path.exists(model_path):
            raise FileNotFoundError("モデルファイルが存在しません。")
        self.policy_net = PolicyNet(5, 4)
        self.policy_net.load_state_dict(torch.load(model_path))
        self.policy_net.eval()
        self.buffer = deque(maxlen=60)
        self.position = None

    def add(self, time: float, price: float, volume: float) -> str:
        self.buffer.append((time, price, volume))
        if len(self.buffer) < 60:
            return "HOLD"

        df = pd.DataFrame(self.buffer, columns=["Time", "Price", "Volume"])
        df_feat = compute_features(df).iloc[-1]
        obs = np.array([
            df_feat["LogDeltaVolume"],
            df_feat["MA"], df_feat["STD"], df_feat["RSI"], df_feat["ZScore"]
        ], dtype=np.float32)

        with torch.no_grad():
            probs = self.policy_net(torch.tensor(obs).unsqueeze(0))
            action = torch.argmax(probs).item()

        return ["HOLD", "BUY", "SELL", "REPAY"][action]
