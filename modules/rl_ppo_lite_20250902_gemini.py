import os
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Constants
N_TICKS_WARMUP = 60
UNIT_SHARES = 100
SLIPPAGE = 1


# Helper function to calculate features
def calculate_features(df):
    df['ΔVolume'] = df['Volume'].diff().fillna(0)
    df['log_ΔVolume'] = np.log1p(df['ΔVolume'])
    df['MA'] = df['Price'].rolling(window=60).mean()
    df['STD'] = df['Price'].rolling(window=60).std()

    # RSI
    delta = df['Price'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=60).mean()
    avg_loss = loss.rolling(window=60).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Z-score
    df['Z_score'] = (df['Price'] - df['MA']) / df['STD']

    # Fill NaN values after calculation (especially for the first 60 rows)
    df = df.fillna(0)
    return df


# ---
# PPO Agent and Networks
# ---

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.net(state)


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        return self.net(state)


class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, clip_epsilon=0.2):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.memory = []

    def get_action(self, state, exploration=True):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.actor(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        if exploration and np.random.rand() < 0.1:  # epsilon-greedy exploration
            action = torch.tensor(np.random.randint(4))
        return action.item()

    def update(self):
        states = torch.FloatTensor([s for s, _, _, _ in self.memory])
        actions = torch.LongTensor([a for _, a, _, _ in self.memory]).unsqueeze(1)
        rewards = torch.FloatTensor([r for _, _, r, _ in self.memory])
        dones = torch.FloatTensor([d for _, _, _, d in self.memory]).unsqueeze(1)

        # Calculate returns
        returns = []
        R = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d:
                R = 0
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns).unsqueeze(1)

        # PPO update loop
        for _ in range(10):  # PPO inner loop
            values = self.critic(states)
            advantages = returns - values

            old_probs = self.actor(states).gather(1, actions).detach()
            new_probs = self.actor(states).gather(1, actions)

            ratio = new_probs / (old_probs + 1e-8)
            surr1 = ratio * advantages.detach()
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages.detach()

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values, returns)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        self.memory.clear()

    def save(self, path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor.eval()
        self.critic.eval()


# ---
# Trading Environment
# ---

class TradingEnv(gym.Env):
    def __init__(self, df):
        super(TradingEnv, self).__init__()
        self.df = df
        self.state_dim = 5
        self.action_space = spaces.Discrete(4)  # 0:HOLD, 1:BUY, 2:SELL, 3:REPAY
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        self.current_tick = 0
        self.position = None  # None: no position, "long": long position, "short": short position
        self.entry_price = 0
        self.total_profit = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_tick = N_TICKS_WARMUP - 1
        self.position = None
        self.entry_price = 0
        self.total_profit = 0
        initial_state = self._get_state()
        info = {}
        return initial_state, info

    def _get_state(self):
        row = self.df.iloc[self.current_tick]
        return np.array([
            row['log_ΔVolume'],
            row['MA'],
            row['RSI'],
            row['Z_score'],
            float(self.position is not None)  # 1 if position exists, 0 otherwise
        ])

    def step(self, action):
        # Apply trading constraints and logic
        if self.current_tick < N_TICKS_WARMUP - 1:
            action = 0  # HOLD during warmup

        current_price = self.df.iloc[self.current_tick]['Price']
        reward = 0

        # Apply trading constraints
        # Cannot BUY or SELL if a position is held
        if self.position is not None and (action == 1 or action == 2):
            action = 0
            # Cannot REPAY if no position is held
        if self.position is None and action == 3:
            action = 0

            # Execute action
        if action == 1:  # BUY
            self.position = 'long'
            self.entry_price = current_price + SLIPPAGE
        elif action == 2:  # SELL (short)
            self.position = 'short'
            self.entry_price = current_price - SLIPPAGE
        elif action == 3:  # REPAY
            if self.position == 'long':
                exit_price = current_price - SLIPPAGE
                profit = (exit_price - self.entry_price) * UNIT_SHARES
                reward = profit
            elif self.position == 'short':
                exit_price = current_price + SLIPPAGE
                profit = (self.entry_price - exit_price) * UNIT_SHARES
                reward = profit
            self.total_profit += profit
            self.position = None
        else:  # HOLD
            # Reward for holding a profitable position
            if self.position == 'long':
                reward = 0.001 * (current_price - self.entry_price) * UNIT_SHARES
            elif self.position == 'short':
                reward = 0.001 * (self.entry_price - current_price) * UNIT_SHARES

        self.current_tick += 1

        # Calculate next state and check for termination
        if self.current_tick < len(self.df):
            next_state = self._get_state()
            done = False
        else:
            # Force close any open position at the end of the day
            if self.position == 'long':
                exit_price = current_price - SLIPPAGE
                profit = (exit_price - self.entry_price) * UNIT_SHARES
                self.total_profit += profit
            elif self.position == 'short':
                exit_price = current_price + SLIPPAGE
                profit = (self.entry_price - exit_price) * UNIT_SHARES
                self.total_profit += profit
            self.position = None

            # For the final step, return a dummy state and set done to True
            next_state = np.zeros(self.state_dim)
            done = True

        info = {
            'action_str': ["HOLD", "BUY", "SELL", "REPAY"][action],
            'profit_this_tick': reward,
            'total_profit': self.total_profit,
            'position': self.position
        }

        return next_state, reward, done, False, info

# ---
# Trainer Class (for PC1)
# ---

class Trainer:
    def __init__(self, model_path="policy.pth"):
        self.model_path = model_path
        self.state_dim = 5
        self.action_dim = 4
        self.agent = PPOAgent(self.state_dim, self.action_dim)
        if os.path.exists(self.model_path):
            self.agent.load(self.model_path)

    def train(self, df: pd.DataFrame) -> pd.DataFrame:
        df = calculate_features(df.copy())
        env = TradingEnv(df)

        df_transaction = pd.DataFrame(columns=['Time', 'Price', 'Action', 'Profit', 'Total_Profit'])

        state, info = env.reset()
        done = False

        actions_map = {0: "HOLD", 1: "BUY", 2: "SELL", 3: "REPAY"}

        while not done:
            action = self.agent.get_action(state, exploration=True)
            next_state, reward, done, _, info = env.step(action)
            self.agent.memory.append((state, action, reward, done))

            # Record transaction
            tick_data = df.iloc[env.current_tick - 1]
            df_transaction.loc[len(df_transaction)] = [
                tick_data['Time'],
                tick_data['Price'],
                info['action_str'],
                info['profit_this_tick'],
                info['total_profit']
            ]

            state = next_state

        self.agent.update()  # Update the model at the end of the episode
        self.agent.save(self.model_path)  # Save the updated model

        return df_transaction


# ---
# TradingSimulator Class (for PC2)
# ---

class TradingSimulator:
    def __init__(self, model_path="policy.pth"):
        self.model_path = model_path
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

        self.state_dim = 5
        self.action_dim = 4
        self.agent = PPOAgent(self.state_dim, self.action_dim)
        self.agent.load(self.model_path)

        self.df_ticks = deque(maxlen=N_TICKS_WARMUP)
        self.position = None
        self.is_warmed_up = False

    def add(self, time: float, price: float, volume: float) -> str:
        self.df_ticks.append({
            'Time': time,
            'Price': price,
            'Volume': volume
        })

        if len(self.df_ticks) < N_TICKS_WARMUP:
            return "HOLD"

        if not self.is_warmed_up:
            df = pd.DataFrame(list(self.df_ticks))
            self.df = calculate_features(df)
            self.is_warmed_up = True

        current_tick = len(self.df_ticks) - 1
        current_data = self.df_ticks[current_tick]

        # Update features with new tick data
        delta_vol = current_data['Volume'] - self.df_ticks[current_tick - 1]['Volume']
        self.df.loc[current_tick, 'ΔVolume'] = delta_vol
        self.df.loc[current_tick, 'log_ΔVolume'] = np.log1p(delta_vol)
        self.df['MA'] = self.df['Price'].rolling(window=60).mean()
        self.df['STD'] = self.df['Price'].rolling(window=60).std()

        # RSI
        delta = self.df['Price'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=60).mean()
        avg_loss = loss.rolling(window=60).mean()
        rs = avg_gain / avg_loss
        self.df['RSI'] = 100 - (100 / (1 + rs))

        # Z-score
        self.df['Z_score'] = (self.df['Price'] - self.df['MA']) / self.df['STD']

        state = np.array([
            self.df.iloc[-1]['log_ΔVolume'],
            self.df.iloc[-1]['MA'],
            self.df.iloc[-1]['RSI'],
            self.df.iloc[-1]['Z_score'],
            float(self.position is not None)
        ])

        action_int = self.agent.get_action(state, exploration=False)
        action_str = ["HOLD", "BUY", "SELL", "REPAY"][action_int]

        # Enforce trading constraints
        if self.position is not None and (action_str == "BUY" or action_str == "SELL"):
            action_str = "HOLD"
        if self.position is None and action_str == "REPAY":
            action_str = "HOLD"

        if action_str == "BUY":
            self.position = "long"
        elif action_str == "SELL":
            self.position = "short"
        elif action_str == "REPAY":
            self.position = None

        return action_str


# ---
# Example of using the classes
# ---
if __name__ == "__main__":
    # This is for testing the Trainer class (PC1)
    # The actual get_excel_sheet and AppRes functions need to be implemented
    # as they are specific to your application environment.

    # Dummy data generation for demonstration
    from datetime import datetime, timedelta


    def generate_dummy_data(start_time, n_ticks=19500):
        times = [start_time + timedelta(seconds=i) for i in range(n_ticks)]
        prices = [10000 + 50 * np.sin(i / 1000) + np.random.randn() * 10 for i in range(n_ticks)]
        volumes = [10000 + 100 * i + np.random.randint(0, 500) for i in range(n_ticks)]
        df = pd.DataFrame({
            'Time': times,
            'Price': prices,
            'Volume': volumes
        })
        return df


    # Learning loop
    print("--- Starting Trainer (PC1) simulation ---")
    trainer = Trainer()
    df_transaction_list = []

    for epoch in range(5):
        print(f"Epoch {epoch + 1}")
        # Assuming you have a list of daily tick data DataFrames
        dummy_df_day1 = generate_dummy_data(datetime(2025, 9, 1, 9, 0, 0))
        dummy_df_day2 = generate_dummy_data(datetime(2025, 9, 2, 9, 0, 0))

        df_transaction = trainer.train(dummy_df_day1)
        print(f"Day 1 total profit: {df_transaction['Profit'].sum():.2f}")
        df_transaction_list.append(df_transaction)

        df_transaction = trainer.train(dummy_df_day2)
        print(f"Day 2 total profit: {df_transaction['Profit'].sum():.2f}")
        df_transaction_list.append(df_transaction)

    # Example of saving learning curve
    df_lc = pd.DataFrame({
        "Epoch": range(len(df_transaction_list)),
        "Data": [f"Day_{i % 2 + 1}" for i in range(len(df_transaction_list))],
        "Profit": [df['Profit'].sum() for df in df_transaction_list]
    })
    print("\nLearning Curve Data:")
    print(df_lc)

    # This is for testing the TradingSimulator class (PC2)
    print("\n--- Starting TradingSimulator (PC2) simulation ---")
    try:
        simulator = TradingSimulator()

        # Simulate real-time tick data feed
        dummy_df_today = generate_dummy_data(datetime(2025, 9, 3, 9, 0, 0), n_ticks=100)

        for index, row in dummy_df_today.iterrows():
            action = simulator.add(row['Time'], row['Price'], row['Volume'])
            print(f"Time: {row['Time']}, Price: {row['Price']:.2f}, Action: {action}")

    except FileNotFoundError as e:
        print(f"Error: {e}. Please run the Trainer simulation first to create the model file.")
