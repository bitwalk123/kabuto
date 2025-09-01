import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import gymnasium as gym


# ============================================================
#  ユーティリティ: テクニカル指標の計算
# ============================================================

def compute_features(prices, volumes, window=60):
    """最新ティックまでの特徴量を返す"""
    if len(prices) < window + 1:
        return None

    arr_p = np.array(prices[-window:])
    arr_v = np.diff(np.array(volumes[-(window + 1):]))  # ΔVolume

    ma = arr_p.mean()
    std = arr_p.std() if arr_p.std() > 1e-6 else 1.0
    rsi = calc_rsi(arr_p, n=window)
    zscore = (arr_p[-1] - ma) / std
    dvol = np.log1p(arr_v[-1])

    return np.array([arr_p[-1], dvol, ma, std, rsi, zscore], dtype=np.float32)


def calc_rsi(prices, n=60):
    deltas = np.diff(prices)
    up = deltas[deltas > 0].sum() / n
    down = -deltas[deltas < 0].sum() / n
    rs = up / down if down != 0 else 0
    return 100. - (100. / (1. + rs))


# ============================================================
#  PPO ネットワーク
# ============================================================

class PolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )
        self.logits = nn.Linear(64, output_dim)

    def forward(self, x):
        h = self.fc(x)
        return self.logits(h)


class ValueNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )
        self.v = nn.Linear(64, 1)

    def forward(self, x):
        h = self.fc(x)
        return self.v(h)


# ============================================================
#  環境クラス
# ============================================================

class TradingEnv(gym.Env):
    def __init__(self, df, unit=100, slippage=1, window=60):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.unit = unit
        self.slippage = slippage
        self.window = window

        self.action_space = gym.spaces.Discrete(4)  # HOLD, BUY, SELL, REPAY
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.i = 0
        self.pos = 0  # 0=ノーポジ, 1=ロング, -1=ショート
        self.entry_price = 0
        self.done = False
        self.total_profit = 0.0
        self.profits = []

        self.prices = []
        self.volumes = []
        obs, _ = self._step_next()
        return obs, {}

    def _step_next(self):
        row = self.df.iloc[self.i]
        self.prices.append(row["Price"])
        self.volumes.append(row["Volume"])
        feat = compute_features(self.prices, self.volumes, self.window)
        self.i += 1
        if feat is None:
            return np.zeros(6, dtype=np.float32), False
        return feat, self.i >= len(self.df)

    def step(self, action):
        reward = 0.0
        row = self.df.iloc[self.i - 1]
        price = row["Price"]

        if action == 1 and self.pos == 0:  # BUY
            self.pos = 1
            self.entry_price = price + self.slippage
        elif action == 2 and self.pos == 0:  # SELL
            self.pos = -1
            self.entry_price = price - self.slippage
        elif action == 3 and self.pos != 0:  # REPAY
            if self.pos == 1:
                exit_price = price - self.slippage
                profit = (exit_price - self.entry_price) * self.unit
            else:
                exit_price = price + self.slippage
                profit = (self.entry_price - exit_price) * self.unit
            reward += profit
            self.total_profit += profit
            self.profits.append(profit)
            self.pos = 0
            self.entry_price = 0

        # 含み益を報酬に追加
        if self.pos != 0:
            if self.pos == 1:
                unreal = (price - self.entry_price) * self.unit
            else:
                unreal = (self.entry_price - price) * self.unit
            reward += 0.001 * unreal

        obs, end = self._step_next()
        if end:
            if self.pos != 0:  # 強制決済
                if self.pos == 1:
                    exit_price = price - self.slippage
                    profit = (exit_price - self.entry_price) * self.unit
                else:
                    exit_price = price + self.slippage
                    profit = (self.entry_price - exit_price) * self.unit
                reward += profit
                self.total_profit += profit
                self.profits.append(profit)
                self.pos = 0
            self.done = True

        return obs, reward, self.done, False, {}


# ============================================================
#  PPO エージェント
# ============================================================

class PPOAgent:
    def __init__(self, input_dim, action_dim, lr=1e-3, gamma=0.99, eps_clip=0.2):
        self.policy = PolicyNet(input_dim, action_dim)
        self.value = ValueNet(input_dim)
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()), lr=lr
        )
        self.gamma = gamma
        self.eps_clip = eps_clip

    def select_action(self, state, epsilon=0.1):
        state = torch.FloatTensor(state).unsqueeze(0)
        logits = self.policy(state)
        probs = torch.softmax(logits, dim=-1).detach().numpy()[0]

        # NaN / inf ガード
        probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
        if probs.sum() <= 0:
            probs = np.ones_like(probs) / len(probs)
        else:
            probs = probs / probs.sum()

        # ε-greedy
        if np.random.rand() < epsilon:
            action = np.random.choice(len(probs))
        else:
            action = np.random.choice(len(probs), p=probs)
        return action

    def update(self, memory):
        states = torch.FloatTensor(np.array(memory["states"]))
        actions = torch.LongTensor(memory["actions"])
        rewards = memory["rewards"]
        dones = memory["dones"]
        next_states = torch.FloatTensor(np.array(memory["next_states"]))

        returns, advs = self.compute_gae(states, rewards, dones, next_states)

        old_logits = self.policy(states)
        old_probs = torch.softmax(old_logits, dim=-1).gather(1, actions.unsqueeze(1)).detach()

        for _ in range(4):
            logits = self.policy(states)
            probs = torch.softmax(logits, dim=-1).gather(1, actions.unsqueeze(1))
            ratio = probs / old_probs

            advs_t = torch.FloatTensor(advs).unsqueeze(1)
            surr1 = ratio * advs_t
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advs_t
            actor_loss = -torch.min(surr1, surr2).mean()

            values = self.value(states).squeeze()
            returns_t = torch.FloatTensor(returns)
            critic_loss = (returns_t - values).pow(2).mean()

            loss = actor_loss + 0.5 * critic_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def compute_gae(self, states, rewards, dones, next_states, lam=0.95):
        values = self.value(states).squeeze().detach().numpy()
        next_values = self.value(next_states).squeeze().detach().numpy()
        deltas = np.array(rewards) + self.gamma * next_values * (1 - np.array(dones)) - values
        advs, gae = [], 0
        for d in reversed(deltas):
            gae = d + self.gamma * lam * gae
            advs.insert(0, gae)
        returns = values + advs
        return returns, advs

    def save(self, path="policy.pth"):
        torch.save({"policy": self.policy.state_dict(),
                    "value": self.value.state_dict()}, path)

    def load(self, path="policy.pth"):
        data = torch.load(path)
        self.policy.load_state_dict(data["policy"])
        self.value.load_state_dict(data["value"])


# ============================================================
#  Trainer クラス（学習用）
# ============================================================

class Trainer:
    def __init__(self, model_path="policy.pth"):
        self.model_path = model_path
        self.agent = PPOAgent(input_dim=6, action_dim=4)

        if os.path.exists(self.model_path):
            self.agent.load(self.model_path)

    def train(self, df):
        env = TradingEnv(df)
        memory = {"states": [], "actions": [], "rewards": [], "dones": [], "next_states": []}
        obs, _ = env.reset()
        done = False

        records = []

        while not done:
            action = self.agent.select_action(obs, epsilon=0.1)
            next_obs, reward, done, _, _ = env.step(action)

            memory["states"].append(obs)
            memory["actions"].append(action)
            memory["rewards"].append(reward)
            memory["dones"].append(done)
            memory["next_states"].append(next_obs)

            obs = next_obs

            records.append({
                "Time": df.iloc[env.i - 1]["Time"],
                "Price": df.iloc[env.i - 1]["Price"],
                "Action": ["HOLD", "BUY", "SELL", "REPAY"][action],
                "Profit": env.profits[-1] if env.profits else 0
            })

        self.agent.update(memory)
        self.agent.save(self.model_path)

        return pd.DataFrame(records)


# ============================================================
#  TradingSimulator クラス（推論専用）
# ============================================================

class TradingSimulator:
    def __init__(self, model_path="policy.pth", window=60):
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model not found")
        self.agent = PPOAgent(input_dim=6, action_dim=4)
        self.agent.load(model_path)
        self.window = window
        self.prices = []
        self.volumes = []
        self.pos = 0

    def add(self, time: float, price: float, volume: float) -> str:
        self.prices.append(price)
        self.volumes.append(volume)
        feat = compute_features(self.prices, self.volumes, self.window)
        if feat is None:
            return "HOLD"
        action = self.agent.select_action(feat, epsilon=0.0)  # 推論時は ε=0
        return ["HOLD", "BUY", "SELL", "REPAY"][action]
