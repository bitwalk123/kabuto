import numpy as np
import pandas as pd
import os
import gymnasium as gym
from gymnasium import spaces
import torch

from modules.ppo_model import PPOAgent


# 環境の定義（Gymnasium準拠）
class TradingEnv(gym.Env):
    def __init__(self, df, window_size=60):
        super(TradingEnv, self).__init__()
        self.df = df
        self.window_size = window_size
        self.n_features = 5  # ΔVolume, MA, STD, RSI, Z-score
        self.action_space = spaces.Discrete(4)  # HOLD, BUY, SELL, REPAY
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_features,), dtype=np.float32)
        self.reset()

    def _calculate_features(self, step):
        if step < self.window_size:
            return np.zeros(self.n_features)

        window = self.df.iloc[step - self.window_size + 1: step + 1]

        # 1. ΔVolume
        delta_volume = window['Volume'].diff().fillna(0).iloc[-1]
        log_delta_volume = np.log1p(delta_volume) if delta_volume > 0 else 0

        # 2. MA
        ma = window['Price'].mean()

        # 3. STD
        std = window['Price'].std()

        # 4. RSI
        gains = np.maximum(window['Price'].diff(), 0)
        losses = np.maximum(-window['Price'].diff(), 0)
        avg_gain = gains.mean()
        avg_loss = losses.mean()
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        # 5. Z-score
        z_score = (window['Price'].iloc[-1] - ma) / (std if std > 0 else 1)

        return np.array([log_delta_volume, ma, std, rsi, z_score])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.total_profit = 0
        self.position = 0  # 0:なし, 1:ロング, -1:ショート
        self.entry_price = 0
        self.entry_step = 0
        self.terminated = False
        self.truncated = False

        initial_state = self._calculate_features(self.current_step)
        return initial_state, {}

    def step(self, action, force_close=False):
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        reward = 0
        info = {'position': self.position}
        current_price = self.df['Price'].iloc[self.current_step]
        slip = 1

        # HOLD
        if action == 0 and not force_close:
            if self.position != 0:
                unrealized_profit = (current_price - self.entry_price) * self.position * 100
                reward += unrealized_profit * 0.05

        # BUY (新規買い)
        elif action == 1 and self.position == 0 and self.current_step >= self.window_size and not force_close:
            self.position = 1
            self.entry_price = current_price + slip

        # SELL (新規売り)
        elif action == 2 and self.position == 0 and self.current_step >= self.window_size and not force_close:
            self.position = -1
            self.entry_price = current_price - slip

        # REPAY (返済)
        elif (action == 3 and self.position != 0) or force_close:
            exit_price = current_price + slip * self.position
            profit = (exit_price - self.entry_price) * (-self.position) * 100
            self.total_profit += profit
            reward = profit
            self.position = 0
            self.entry_price = 0

        next_state = self._calculate_features(self.current_step)

        self.terminated = done
        return next_state, reward, self.terminated, self.truncated, info


# Trainerクラス: 学習用
class Trainer:
    def __init__(self, model_path='policy.pth'):
        self.model_path = model_path
        self.agent = PPOAgent(input_dim=5, output_dim=4)
        self.action_map_str = {0: "HOLD", 1: "BUY", 2: "SELL", 3: "REPAY"}

    def train(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Price'] = df['Price'].astype(float)
        df['Volume'] = df['Volume'].astype(float)

        env = TradingEnv(df)

        states, actions, old_log_probs, rewards, dones, next_states = [], [], [], [], [], []
        df_transaction = pd.DataFrame(columns=['Time', 'Price', 'Action', 'Profit'])

        state, _ = env.reset()

        for i in range(len(df) - 1):
            action, log_prob, value = self.agent.get_action_and_value(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            states.append(state)
            actions.append(action)
            old_log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(terminated)
            next_states.append(next_state)

            # トランザクションを記録
            if action in [1, 2, 3]:  # BUY, SELL, REPAY
                if action == 3:
                    if env.position == 0:
                        df_transaction.loc[len(df_transaction)] = [df.iloc[i]['Time'], df.iloc[i]['Price'],
                                                                   self.action_map_str[action], reward]
                else:
                    df_transaction.loc[len(df_transaction)] = [df.iloc[i]['Time'], df.iloc[i]['Price'],
                                                               self.action_map_str[action], np.nan]

            state = next_state
            if terminated:
                break

        # 最終行で建玉があれば強制返済
        if env.position != 0:
            final_price = df['Price'].iloc[-1]
            slip = 1
            exit_price = final_price + slip * env.position
            profit = (exit_price - env.entry_price) * (-env.position) * 100
            env.total_profit += profit
            rewards[-1] += profit
            dones[-1] = 1
            # 最終トランザクションを記録
            df_transaction.loc[len(df_transaction)] = [df.iloc[-1]['Time'], final_price, "REPAY (Force Close)", profit]

        self.agent.update(states, actions, old_log_probs, rewards, dones, next_states)
        self.agent.save_model(self.model_path)

        return df_transaction


# TradingSimulatorクラス: 推論用
class TradingSimulator:
    def __init__(self, model_path='policy.pth'):
        self.model_path = model_path
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}. Please train a model first.")

        self.agent = PPOAgent(input_dim=5, output_dim=4)
        self.agent.load_model(self.model_path)
        self.agent.policy.eval()  # 推論モード
        self.agent.value.eval()

        self.tick_data = pd.DataFrame(columns=['Time', 'Price', 'Volume'])
        self.current_step = 0
        self.window_size = 60

    def _calculate_features(self):
        if len(self.tick_data) < self.window_size:
            return np.zeros(5)

        window = self.tick_data.tail(self.window_size)

        # 1. ΔVolume
        delta_volume = window['Volume'].diff().fillna(0).iloc[-1]
        log_delta_volume = np.log1p(delta_volume) if delta_volume > 0 else 0

        # 2. MA
        ma = window['Price'].mean()

        # 3. STD
        std = window['Price'].std()

        # 4. RSI
        gains = np.maximum(window['Price'].diff(), 0)
        losses = np.maximum(-window['Price'].diff(), 0)
        avg_gain = gains.mean()
        avg_loss = losses.mean()
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        # 5. Z-score
        z_score = (window['Price'].iloc[-1] - ma) / (std if std > 0 else 1)

        return np.array([log_delta_volume, ma, std, rsi, z_score])

    def add(self, time: float, price: float, volume: float) -> str:
        self.tick_data.loc[len(self.tick_data)] = [time, price, volume]
        self.current_step += 1

        # ウォームアップ期間
        if self.current_step <= self.window_size:
            return "HOLD"

        features = self._calculate_features()
        state_tensor = torch.FloatTensor(features).unsqueeze(0)

        with torch.no_grad():
            probs = self.agent.policy(state_tensor)
            action = torch.argmax(probs).item()

        action_map = {0: "HOLD", 1: "BUY", 2: "SELL", 3: "REPAY"}
        return action_map.get(action, "HOLD")


if __name__ == '__main__':
    # サンプルデータの生成 (実際は証券会社から取得)
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'Time': np.arange(1, 10001),
        'Price': np.random.normal(1000, 5, 10000).cumsum() + 1000,
        'Volume': np.random.randint(100, 1000, 10000).cumsum()
    })

    # PC2 (引け後用): 学習の実行
    print("--- Learning ---")
    trainer = Trainer()
    transaction_log = trainer.train(sample_data)
    print("Training finished.")
    print("Transactions:")
    print(transaction_log)
    print(f"\nTotal Profit: {transaction_log['Profit'].sum():.2f} JPY")

    # PC1 (ザラ場用): 推論の実行
    print("\n--- Inference ---")
    try:
        simulator = TradingSimulator()

        # 証券会社からリアルタイムデータが来るのをシミュレート
        print("Simulating real-time data feed...")
        for idx, row in sample_data.iterrows():
            action_str = simulator.add(row['Time'], row['Price'], row['Volume'])
            # 1ティックごとに処理
            if idx % 100 == 0:
                print(f"Time: {row['Time']}, Price: {row['Price']:.2f}, Action: {action_str}")
            if idx > 1000:  # 1000ティックで停止
                break

    except FileNotFoundError as e:
        print(e)