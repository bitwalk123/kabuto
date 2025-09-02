import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import random


class Trainer:
    def __init__(self, model_path="policy.pth", learning_rate=1e-4, gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2,
                 c1=0.5, c2=0.01):
        self.model_path = model_path
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.c1 = c1  # Value function coefficient
        self.c2 = c2  # Entropy coefficient
        self.n_actions = 4  # BUY, SELL, HOLD, REPAY
        self.n_features = 5  # ΔVolume, MA, STD, RSI, Zscore
        self.unit_size = 100
        self.tick_size = 1  # 1ティックのスリッページ

        # PPOモデル
        self.policy_net = self.PolicyNetwork(self.n_features, self.n_actions)
        self.value_net = self.ValueNetwork(self.n_features)

        # Optimizer
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.learning_rate)

        self._load_model()

    class PolicyNetwork(nn.Module):
        def __init__(self, n_features, n_actions):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_features, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, n_actions)
            )

        def forward(self, x):
            return torch.softmax(self.net(x), dim=-1)

    class ValueNetwork(nn.Module):
        def __init__(self, n_features):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_features, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )

        def forward(self, x):
            return self.net(x)

    def _load_model(self):
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path)
            self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
            self.value_net.load_state_dict(checkpoint['value_state_dict'])
            print("PPO model loaded.")
        else:
            print("No existing PPO model found. Training from scratch.")

    def _save_model(self):
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
        }, self.model_path)
        print(f"PPO model saved to {self.model_path}")

    def _preprocess_data(self, df):
        # 出来高の増分
        df["ΔVolume"] = df["Volume"].diff().fillna(0)
        df["log_ΔVolume"] = np.log1p(df["ΔVolume"])

        # 移動平均、標準偏差
        df["MA"] = df["Price"].rolling(window=60).mean()
        df["STD"] = df["Price"].rolling(window=60).std()

        # RSI
        delta = df["Price"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=60).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=60).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        # Zスコア
        df["Zscore"] = (df["Price"] - df["MA"]) / df["STD"]

        df = df.fillna(0)  # 最初の60ティックは0で埋める

        features = df[["log_ΔVolume", "MA", "STD", "RSI", "Zscore"]].values

        return features

    def train(self, df):
        features = self._preprocess_data(df)

        # 環境設定
        position = 0  # 0:なし, 1:ロング, -1:ショート
        entry_price = 0
        total_profit = 0

        # ログ用データフレーム
        df_transaction = pd.DataFrame(columns=["Time", "Price", "Action", "Profit"])

        # 経験バッファ
        states, actions, rewards, old_log_probs, values = [], [], [], [], []

        # 各ティックで学習
        for i in range(len(df)):
            time = df.iloc[i]["Time"]
            price = df.iloc[i]["Price"]
            reward = 0  # 💡 UnboundLocalError 修正: ループの最初に reward を初期化

            # ウォームアップ期間
            if i < 60:
                action_str = "HOLD"
                action_idx = 0
                state = features[i]
                log_prob = 0.0  # 💡 UnboundLocalError 修正: ダミーの値を代入
                value = 0.0  # 💡 UnboundLocalError 修正: ダミーの値を代入

            else:
                state = features[i]
                state_tensor = torch.FloatTensor(state)

                with torch.no_grad():
                    probs = self.policy_net(state_tensor)
                    dist = torch.distributions.Categorical(probs)

                # 制約条件の適用
                if position != 0:
                    probs_constrained = probs.clone()
                    probs_constrained[0] += probs_constrained[3]
                    probs_constrained[1] = 0
                    probs_constrained[2] = 0
                    probs_constrained[3] = 0
                    if probs_constrained.sum() > 0:
                        probs_constrained /= probs_constrained.sum()
                    dist = torch.distributions.Categorical(probs_constrained)

                # アクションのサンプリング
                action_idx = dist.sample().item()
                log_prob = dist.log_prob(torch.tensor(action_idx))
                value = self.value_net(state_tensor).item()

                # ε-greedy的な探索
                if random.random() < 0.1:
                    action_idx = random.choice([0, 1, 2, 3])

                if action_idx == 1 and position == 0:  # BUY
                    action_str = "BUY"
                    entry_price = price + self.tick_size
                    position = 1
                elif action_idx == 2 and position == 0:  # SELL
                    action_str = "SELL"
                    entry_price = price - self.tick_size
                    position = -1
                elif action_idx == 3 and position != 0:  # REPAY
                    action_str = "REPAY"
                    if position == 1:
                        profit = (price - self.tick_size - entry_price) * self.unit_size
                    else:
                        profit = (entry_price - (price + self.tick_size)) * self.unit_size
                    total_profit += profit
                    reward = profit * 0.01
                    position = 0
                    entry_price = 0
                else:  # HOLD
                    action_str = "HOLD"
                    if position == 1:
                        reward = (price - entry_price) * self.unit_size * 0.001
                    elif position == -1:
                        reward = (entry_price - price) * self.unit_size * 0.001

            # 💡 IndexError 修正: 全てのループでリストに追加
            states.append(state)
            actions.append(action_idx)
            old_log_probs.append(log_prob.item() if isinstance(log_prob, torch.Tensor) else log_prob)
            values.append(value)
            rewards.append(reward)

            # ログ記録
            df_transaction.loc[len(df_transaction)] = [time, price, action_str, total_profit]

        # 最終行での強制返済処理（ループ外で実行）
        if position != 0:
            last_price = df.iloc[-1]["Price"]
            if position == 1:
                profit = (last_price - self.tick_size - entry_price) * self.unit_size
            else:
                profit = (entry_price - (last_price + self.tick_size)) * self.unit_size

            # 最後のrewardを強制決済の報酬で上書き
            rewards[-1] = profit * 0.01

            df_transaction.loc[len(df_transaction)] = [df.iloc[-1]["Time"], last_price, "REPAY_FORCE",
                                                       total_profit + profit]

        # 💡 IndexError 修正: dones リストを最後に一括で作成
        dones = [False] * (len(rewards) - 1) + [True]

        # PPO学習
        # 💡 IndexError 修正: next_states を引数から削除
        self._update_ppo(states, actions, rewards, dones, old_log_probs, values)

        self._save_model()

        return df_transaction

    def _update_ppo(self, states, actions, rewards, next_states, dones, old_log_probs, values):
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        old_log_probs = torch.FloatTensor(old_log_probs)
        values = torch.FloatTensor(values)

        # Generalized Advantage Estimation (GAE)
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        for i in reversed(range(len(rewards))):
            if i < len(rewards) - 1:
                delta = rewards[i] + self.gamma * values[i + 1] * (1 - dones[i]) - values[i]
            else:
                delta = rewards[i] - values[i]
            last_gae = delta + self.gamma * self.gae_lambda * last_gae * (1 - dones[i])
            advantages[i] = last_gae

        returns = advantages + values

        # PPO更新
        for _ in range(10):  # 複数エポック学習
            probs = self.policy_net(states)
            dist = torch.distributions.Categorical(probs)
            log_probs = dist.log_prob(actions)

            # PPOクリッピング
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages

            # 方策損失
            policy_loss = -torch.min(surr1, surr2).mean()

            # 価値損失
            value_loss = (returns - self.value_net(states).squeeze()).pow(2).mean()

            # エントロピー損失
            entropy_loss = dist.entropy().mean()

            # 全体損失
            total_loss = policy_loss + self.c1 * value_loss - self.c2 * entropy_loss

            # 最適化
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            total_loss.backward()
            self.policy_optimizer.step()
            self.value_optimizer.step()

        print(f"PPO training done. Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}")


class TradingSimulator:
    def __init__(self, model_path="policy.pth"):
        self.model_path = model_path
        self.n_features = 5
        self.n_actions = 4
        self.tick_data = []
        self.position = 0  # 0:なし, 1:ロング, -1:ショート

        # モデルの読み込み
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Error: Model file not found at {self.model_path}")

        self.policy_net = self.PolicyNetwork(self.n_features, self.n_actions)
        checkpoint = torch.load(self.model_path)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.policy_net.eval()

    class PolicyNetwork(nn.Module):
        def __init__(self, n_features, n_actions):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_features, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, n_actions)
            )

        def forward(self, x):
            return torch.softmax(self.net(x), dim=-1)

    def _preprocess_data(self, df):
        if len(df) < 60:
            return None

        df["ΔVolume"] = df["Volume"].diff().fillna(0)
        df["log_ΔVolume"] = np.log1p(df["ΔVolume"])

        df["MA"] = df["Price"].rolling(window=60).mean()
        df["STD"] = df["Price"].rolling(window=60).std()

        delta = df["Price"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=60).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=60).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        df["Zscore"] = (df["Price"] - df["MA"]) / df["STD"]

        features = df[["log_ΔVolume", "MA", "STD", "RSI", "Zscore"]].iloc[-1].values

        return features

    def add(self, time: float, price: float, volume: float) -> str:
        self.tick_data.append({"Time": time, "Price": price, "Volume": volume})
        df_ticks = pd.DataFrame(self.tick_data).tail(60)  # 直近60ティックを保持

        # ウォームアップ期間
        if len(self.tick_data) < 60:
            return "HOLD"

        state = self._preprocess_data(df_ticks)
        state_tensor = torch.FloatTensor(state)

        # 推論
        with torch.no_grad():
            probs = self.policy_net(state_tensor)

        # 制約条件の適用
        if self.position != 0:
            probs_constrained = probs.clone()
            probs_constrained[0] += probs_constrained[3]
            probs_constrained[1] = 0
            probs_constrained[2] = 0
            probs_constrained[3] = 0
            if probs_constrained.sum() > 0:
                probs_constrained /= probs_constrained.sum()

            action_idx = torch.argmax(probs_constrained).item()
        else:
            action_idx = torch.argmax(probs).item()

        # アクションの実行と建玉の更新
        if action_idx == 1 and self.position == 0:
            self.position = 1
            return "BUY"
        elif action_idx == 2 and self.position == 0:
            self.position = -1
            return "SELL"
        elif action_idx == 3 and self.position != 0:
            self.position = 0
            return "REPAY"
        else:
            return "HOLD"
