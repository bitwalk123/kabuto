import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium import spaces

# モデルファイルのパス
MODEL_PATH = "policy_gemini.pth"


# -----------------------------------------------------------------------------
# PPO エージェントの実装
# -----------------------------------------------------------------------------
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)


class ValueNet(nn.Module):
    def __init__(self, state_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, clip_epsilon=0.2):
        self.policy_net = PolicyNet(state_dim, action_dim)
        self.value_net = ValueNet(state_dim)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.action_dim = action_dim

    def save_model(self, path=MODEL_PATH):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
        }, path)

    def load_model(self, path=MODEL_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.value_net.load_state_dict(checkpoint['value_net'])

    def get_action(self, state, exploration=True):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.policy_net(state_tensor)
        if exploration and np.random.rand() < 0.1:  # ε-greedy的な探索
            action = np.random.choice(self.action_dim)
        else:
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample().item()
        return action, action_probs.squeeze(0)[action].item()

    def update(self, states, actions, old_log_probs, returns, advantages):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)

        # Value Loss
        values = self.value_net(states).squeeze(-1)
        value_loss = nn.functional.mse_loss(values, returns)

        # Policy Loss
        action_probs = self.policy_net(states)
        dist = torch.distributions.Categorical(action_probs)
        new_log_probs = dist.log_prob(actions)

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Update
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()


# -----------------------------------------------------------------------------
# 特徴量エンジニアリング
# -----------------------------------------------------------------------------
def featurize_data(df):
    df_features = pd.DataFrame(index=df.index)

    # 出来高の増分 ΔVolume
    df_features['delta_volume'] = df['Volume'].diff().fillna(0)
    df_features['log_delta_volume'] = np.log1p(df_features['delta_volume'])

    # 移動平均 (MA)
    df_features['ma'] = df['Price'].rolling(window=60).mean().bfill()

    # 標準偏差 (STD)
    df_features['std'] = df['Price'].rolling(window=60).std().bfill()

    # RSI (Relative Strength Index)
    delta = df['Price'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=60).mean().fillna(0)
    avg_loss = loss.rolling(window=60).mean().fillna(0)
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    df_features['rsi'] = 100 - (100 / (1 + rs))
    df_features['rsi'] = df_features['rsi'].bfill()

    # Zスコア
    df_features['z_score'] = (df['Price'] - df_features['ma']) / (df_features['std'].replace(0, 1e-10))

    # 状態の正規化 (特徴量のスケールを揃える)
    features_to_normalize = ['log_delta_volume', 'ma', 'std', 'rsi', 'z_score']
    for col in features_to_normalize:
        std_val = df_features[col].std()
        if std_val == 0:
            std_val = 1e-10
        df_features[col] = (df_features[col] - df_features[col].mean()) / std_val

    # 必須のNaN処理
    df_features = df_features.fillna(0)

    # 最終的な特徴量列を明示的に指定して返す
    final_features = ['log_delta_volume', 'ma', 'std', 'rsi', 'z_score']
    return df_features[final_features]


# -----------------------------------------------------------------------------
# TradingSimulator クラス（PC1用: 推論）
# -----------------------------------------------------------------------------
class TradingSimulator:
    def __init__(self, stock_code):
        self.stock_code = stock_code
        self.agent = PPOAgent(state_dim=5, action_dim=4)
        self.agent.load_model()
        self.history = []
        self.position = None  # None: no position, "long": long position, "short": short position
        self.entry_price = 0
        self.tick_count = 0
        self.data_buffer = []

    def get_state(self):
        # バッファから特徴量を計算
        if len(self.data_buffer) < 60:
            return np.zeros(5)

        df_temp = pd.DataFrame(self.data_buffer[-60:], columns=['Time', 'Price', 'Volume'])
        features_df = featurize_data(df_temp)
        return features_df.iloc[-1].values

    def add(self, time: float, price: float, volume: float) -> str:
        self.tick_count += 1
        self.data_buffer.append([time, price, volume])

        # 60ティックのウォームアップ期間
        if self.tick_count <= 60:
            return "HOLD"

        state = self.get_state()
        action, _ = self.agent.get_action(state, exploration=False)
        action_str = self.map_action_to_string(action)

        # アクションの制限
        if self.position:
            if action_str not in ["HOLD", "REPAY"]:
                action_str = "HOLD"
        else:
            if action_str == "REPAY":
                action_str = "HOLD"

        self.execute_trade(action_str, price)
        return action_str

    def map_action_to_string(self, action_id):
        actions = ["HOLD", "BUY", "SELL", "REPAY"]
        return actions[action_id]

    def execute_trade(self, action_str, current_price):
        profit = 0
        if action_str == "BUY" and not self.position:
            self.position = "long"
            self.entry_price = current_price + 1
        elif action_str == "SELL" and not self.position:
            self.position = "short"
            self.entry_price = current_price - 1
        elif action_str == "REPAY" and self.position == "long":
            exit_price = current_price - 1
            profit = (exit_price - self.entry_price) * 100
            self.position = None
            self.entry_price = 0
        elif action_str == "REPAY" and self.position == "short":
            exit_price = current_price + 1
            profit = (self.entry_price - exit_price) * 100
            self.position = None
            self.entry_price = 0

        self.history.append({
            'Time': self.data_buffer[-1][0],
            'Price': self.data_buffer[-1][1],
            'Action': action_str,
            'Profit': profit,
            'Position': self.position
        })


# -----------------------------------------------------------------------------
# Trainer クラス（PC2用: 学習）
# -----------------------------------------------------------------------------
class Trainer:
    def __init__(self):
        self.agent = PPOAgent(state_dim=5, action_dim=4)
        if os.path.exists(MODEL_PATH):
            self.agent.load_model()
        self.episode_data = []

    def train(self, df: pd.DataFrame) -> pd.DataFrame:
        df_features = featurize_data(df)
        states = df_features.values
        df_transaction = pd.DataFrame(columns=['Time', 'Price', 'Action', 'Profit'])

        # 強化学習の環境を定義
        env = TradingEnv(states, df['Price'].values)

        for _ in range(100):  # エピソード数 (1日のデータを複数回学習)
            state, _ = env.reset()
            done = False

            states_buffer, actions_buffer, rewards_buffer, old_log_probs_buffer = [], [], [], []

            while not done:
                action, old_log_prob = self.agent.get_action(state, exploration=True)

                next_state, reward, done, _, info = env.step(action)

                states_buffer.append(state)
                actions_buffer.append(action)
                rewards_buffer.append(reward)
                old_log_probs_buffer.append(old_log_prob)

                state = next_state

                # トランザクションログの記録
                action_str = env.map_action_to_string(action)
                df_transaction = pd.concat([df_transaction, pd.DataFrame([{
                    'Time': df['Time'].iloc[env.current_step - 1],
                    'Price': df['Price'].iloc[env.current_step - 1],
                    'Action': action_str,
                    'Profit': info.get('profit', 0)
                }])], ignore_index=True)

            # PPOの更新処理
            returns = self._compute_returns(rewards_buffer)
            advantages = self._compute_advantages(returns, states_buffer)
            self.agent.update(states_buffer, actions_buffer, old_log_probs_buffer, returns, advantages)

        # 学習後のモデルを保存
        self.agent.save_model()
        return df_transaction

    def _compute_returns(self, rewards):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.agent.gamma * R
            returns.insert(0, R)
        return returns

    def _compute_advantages(self, returns, states):
        values = self.agent.value_net(torch.FloatTensor(states)).squeeze(-1).detach().numpy()
        advantages = np.array(returns) - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages


# -----------------------------------------------------------------------------
# Gymnasium 環境クラス
# -----------------------------------------------------------------------------
class TradingEnv(gym.Env):
    def __init__(self, states, prices):
        super(TradingEnv, self).__init__()
        self.states = states
        self.prices = prices
        self.state_dim = states.shape[1]
        self.action_space = spaces.Discrete(4)  # 0:HOLD, 1:BUY, 2:SELL, 3:REPAY
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)

        self.current_step = 0
        self.position = None
        self.entry_price = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.position = None
        self.entry_price = 0

        # ウォームアップ期間後の最初の状態を返す
        initial_state = self.states[60]
        return initial_state, {}

    def map_action_to_string(self, action_id):
        actions = ["HOLD", "BUY", "SELL", "REPAY"]
        return actions[action_id]

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.states)
        reward = 0
        info = {'profit': 0}

        current_price = self.prices[self.current_step - 1]

        # アクションの制限
        if self.position:
            if action not in [0, 3]:  # HOLD or REPAY
                action = 0
        else:
            if action == 3:  # REPAY
                action = 0

        # 取引ロジックと報酬計算
        if action == 1 and not self.position:  # BUY
            self.position = "long"
            self.entry_price = current_price + 1
        elif action == 2 and not self.position:  # SELL
            self.position = "short"
            self.entry_price = current_price - 1
        elif action == 3 and self.position == "long":  # REPAY (Long)
            exit_price = current_price - 1
            profit = (exit_price - self.entry_price) * 100
            reward = profit
            info['profit'] = profit
            self.position = None
            self.entry_price = 0
        elif action == 3 and self.position == "short":  # REPAY (Short)
            exit_price = current_price + 1
            profit = (self.entry_price - exit_price) * 100
            reward = profit
            info['profit'] = profit
            self.position = None
            self.entry_price = 0

        # 保有中の含み益を報酬に加算
        if self.position == "long":
            profit_temp = (current_price - self.entry_price) * 100
            reward += profit_temp * 0.01  # 含み益の1%を報酬に
        elif self.position == "short":
            profit_temp = (self.entry_price - current_price) * 100
            reward += profit_temp * 0.01

        next_state = self.states[self.current_step] if not done else np.zeros(self.state_dim)
        return next_state, reward, done, False, info