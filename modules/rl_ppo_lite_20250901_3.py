# ppo_trading_sample.py
# 要求環境:
# python==3.13.7, numpy==2.3.2, pandas==2.3.2, torch==2.8.0, gymnasium==1.2.0 (参照のみ)
# CPU 環境前提

import os
import math
import random
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.optim import Adam

# ---------------------------
# ハイパーパラメータ（必要に応じて調整）
# ---------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

DEVICE = torch.device("cpu")
ACTION_MAP = {0: "HOLD", 1: "BUY", 2: "SELL", 3: "REPAY"}

TRADE_UNIT = 100  # 株
SLIPPAGE = 1  # 1ティック
WARMUP = 60  # 60ティック
HOLD_REWARD_FACTOR = 0.05  # 含み益の5%を毎ティック付与

# PPO params
PPO_EPOCHS = 4
PPO_BATCH_SIZE = 64
PPO_CLIP = 0.2
GAMMA = 0.99
LR_POLICY = 3e-4
LR_VALUE = 1e-3
ENTROPY_COEF = 0.01
VALUE_LOSS_COEF = 0.5

# Exploration epsilon for epsilon-greedy during training
EPS_START = 0.2
EPS_END = 0.01

# ---------------------------
# ユーティリティ: 特徴量計算
# ---------------------------
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    df must have columns: Time, Price, Volume (累計)
    Returns df with new columns: dvol_log1p, ma60, std60, rsi60, zscore60
    """
    df = df.copy().reset_index(drop=True)
    # ΔVolume (差分). 初回は 0
    df['dvol'] = df['Volume'].diff().fillna(0).clip(lower=0)
    df['dvol_log1p'] = np.log1p(df['dvol'].values)
    # rolling 60 over Price
    df['ma60'] = df['Price'].rolling(window=WARMUP, min_periods=1).mean()
    df['std60'] = df['Price'].rolling(window=WARMUP, min_periods=1).std(ddof=0).fillna(0)
    # RSI (n=60)
    delta = df['Price'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(WARMUP, min_periods=1).mean()
    roll_down = down.rolling(WARMUP, min_periods=1).mean()
    rs = roll_up / (roll_down + 1e-8)
    df['rsi60'] = 100 - (100 / (1 + rs))
    # z-score
    df['zscore60'] = (df['Price'] - df['ma60']) / (df['std60'] + 1e-8)
    # fill na
    df[['dvol_log1p', 'ma60', 'std60', 'rsi60', 'zscore60']] = df[['dvol_log1p', 'ma60', 'std60', 'rsi60', 'zscore60']].fillna(0)
    return df

# ---------------------------
# ネットワーク
# ---------------------------
class PolicyNet(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128, n_actions: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)


class ValueNet(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

# ---------------------------
# TradingSimulator
# ---------------------------
class TradingSimulator:
    """
    推論用（PC1向け）。
    モデルファイル policy.pth / value.pth が存在しない場合は例外を投げる。
    add(time, price, volume) -> action_str
    """
    def __init__(self, model_path: str = "policy.pth", value_path: str = "value.pth", epsilon: float = 0.0):
        if not os.path.exists(model_path) or not os.path.exists(value_path):
            raise FileNotFoundError(f"Model files not found: {model_path} or {value_path}")
        # feature dimension: dvol_log1p, ma60, std60, rsi60, zscore60, position_flag, entry_price_norm
        self.input_dim = 7
        self.policy = PolicyNet(self.input_dim).to(DEVICE)
        self.value = ValueNet(self.input_dim).to(DEVICE)
        self.policy.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.value.load_state_dict(torch.load(value_path, map_location=DEVICE))
        self.policy.eval()
        self.value.eval()
        self.epsilon = epsilon  # 推論時は 0
        # internal state
        self.prices: List[float] = []
        self.vols: List[float] = []
        self.times: List[float] = []
        self.position: Optional[str] = None  # "LONG" or "SHORT" or None
        self.entry_price: Optional[float] = None  # actual entry price (with slippage)
        self.last_cum_volume: float = 0.0

    def _make_feature(self, idx: int) -> np.ndarray:
        # expects current lists to have at least one element
        price = self.prices[idx]
        # compute features over available history up to idx
        start = max(0, idx - WARMUP + 1)
        window_prices = np.array(self.prices[start:idx+1])
        ma60 = window_prices.mean() if window_prices.size > 0 else price
        std60 = window_prices.std(ddof=0) if window_prices.size > 0 else 0.0
        dvol = max(0.0, self.vols[idx] - (self.vols[idx-1] if idx>=1 else 0.0))
        dvol_log1p = math.log1p(dvol)
        # RSI n=60
        if idx == 0:
            rsi60 = 50.0
        else:
            # compute RSI using price window
            deltas = np.diff(self.prices[max(0, idx - WARMUP):idx+1])
            ups = np.where(deltas > 0, deltas, 0.0)
            downs = np.where(deltas < 0, -deltas, 0.0)
            avg_up = ups.mean() if ups.size>0 else 0.0
            avg_down = downs.mean() if downs.size>0 else 0.0
            rs = avg_up / (avg_down + 1e-8)
            rsi60 = 100 - (100 / (1 + rs))
        z = (price - ma60) / (std60 + 1e-8)
        position_flag = 0.0
        entry_norm = 0.0
        if self.position == "LONG":
            position_flag = 1.0
            entry_norm = (self.entry_price - price) / (price + 1e-8)
        elif self.position == "SHORT":
            position_flag = -1.0
            entry_norm = (self.entry_price - price) / (price + 1e-8)
        feat = np.array([dvol_log1p, ma60, std60, rsi60, z, position_flag, entry_norm], dtype=np.float32)
        return feat

    def add(self, time: float, price: float, volume: float) -> str:
        """
        1ティック毎に呼ぶ。返り値は "HOLD"|"BUY"|"SELL"|"REPAY"
        """
        self.times.append(time)
        self.prices.append(price)
        self.vols.append(volume)
        idx = len(self.prices) - 1

        # Warmup rule
        if idx < WARMUP - 1:
            return "HOLD"

        feat = self._make_feature(idx)
        x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = self.policy(x).cpu().numpy().flatten()
        # epsilon-greedy (inference should set epsilon=0)
        if random.random() < self.epsilon:
            action = random.randrange(4)
        else:
            action = int(np.argmax(probs))

        # enforce action constraints:
        # - If already have LONG, cannot BUY again (no pyramiding). If have SHORT cannot SELL again.
        if action == 1 and self.position == "LONG":
            action = 0
        if action == 2 and self.position == "SHORT":
            action = 0
        # If have no position, cannot REPAY
        if action == 3 and self.position is None:
            action = 0

        action_str = ACTION_MAP[action]

        # Execute action simulation (update position & entry price)
        if action_str == "BUY" and self.position is None:
            # entry price = price + slippage
            entry = price + SLIPPAGE
            self.position = "LONG"
            self.entry_price = entry
        elif action_str == "SELL" and self.position is None:
            entry = price - SLIPPAGE
            self.position = "SHORT"
            self.entry_price = entry
        elif action_str == "REPAY" and self.position is not None:
            # closing position; apply slippage in closing price according to direction
            if self.position == "LONG":
                exit_price = price - SLIPPAGE
                realized = (exit_price - self.entry_price) * TRADE_UNIT
            else:  # SHORT
                exit_price = price + SLIPPAGE
                realized = (self.entry_price - exit_price) * TRADE_UNIT
            # clear position
            self.position = None
            self.entry_price = None
            # (推論時は報酬返却は不要)
        # else HOLD
        return action_str

# ---------------------------
# PPO Trainer
# ---------------------------
class RolloutBuffer:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.values = []

    def clear(self):
        self.__init__()


class Trainer:
    """
    Trainer.train(df) -> df_transaction
    df: DataFrame with Time, Price, Volume (累計)
    """
    def __init__(self, policy_path: str = "policy.pth", value_path: str = "value.pth"):
        self.input_dim = 7
        self.policy = PolicyNet(self.input_dim).to(DEVICE)
        self.value = ValueNet(self.input_dim).to(DEVICE)
        self.optimizer_policy = Adam(self.policy.parameters(), lr=LR_POLICY)
        self.optimizer_value = Adam(self.value.parameters(), lr=LR_VALUE)
        self.buffer = RolloutBuffer()
        self.policy_path = policy_path
        self.value_path = value_path

    def select_action(self, obs: np.ndarray, epsilon: float = 0.0) -> Tuple[int, float, float]:
        x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        probs = self.policy(x)
        m = Categorical(probs)
        # epsilon-greedy (with probability epsilon choose random action)
        if random.random() < epsilon:
            action = random.randrange(4)
            logprob = math.log(1.0 / 4.0 + 1e-8)
        else:
            action = int(m.sample().item())
            logprob = m.log_prob(torch.tensor(action)).item()
        value = self.value(x).item()
        return action, logprob, value

    def compute_gae(self, rewards, values, gamma=GAMMA, lam=0.95):
        # simple discounted returns for now
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        advantages = np.array(returns) - np.array(values)
        return returns, advantages

    def ppo_update(self, obs_arr, actions_arr, old_logprobs_arr, returns_arr, advantages_arr):
        obs_t = torch.tensor(obs_arr, dtype=torch.float32).to(DEVICE)
        actions_t = torch.tensor(actions_arr, dtype=torch.long).to(DEVICE)
        old_logprobs_t = torch.tensor(old_logprobs_arr, dtype=torch.float32).to(DEVICE)
        returns_t = torch.tensor(returns_arr, dtype=torch.float32).to(DEVICE)
        advantages_t = torch.tensor(advantages_arr, dtype=torch.float32).to(DEVICE)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        n = len(obs_arr)
        for _ in range(PPO_EPOCHS):
            # minibatch iterate
            idxs = np.arange(n)
            np.random.shuffle(idxs)
            for start in range(0, n, PPO_BATCH_SIZE):
                mb_idx = idxs[start:start+PPO_BATCH_SIZE]
                mb_obs = obs_t[mb_idx]
                mb_actions = actions_t[mb_idx]
                mb_oldlog = old_logprobs_t[mb_idx]
                mb_returns = returns_t[mb_idx]
                mb_adv = advantages_t[mb_idx]

                probs = self.policy(mb_obs)
                dist = Categorical(probs)
                mb_logprobs = dist.log_prob(mb_actions)
                mb_entropy = dist.entropy().mean()

                ratio = torch.exp(mb_logprobs - mb_oldlog)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean() - ENTROPY_COEF * mb_entropy

                # value loss
                values_pred = self.value(mb_obs).squeeze()
                value_loss = VALUE_LOSS_COEF * (mb_returns - values_pred).pow(2).mean()

                self.optimizer_policy.zero_grad()
                policy_loss.backward()
                self.optimizer_policy.step()

                self.optimizer_value.zero_grad()
                value_loss.backward()
                self.optimizer_value.step()

    def train(self, df: pd.DataFrame, n_updates: int = 1, eps_start: float = EPS_START,
              eps_end: float = EPS_END) -> pd.DataFrame:
        """
        Train policy/value on one day's df. Returns df_transaction with columns Time, Price, Action, Profit.
        Records one transaction row per input Time (every tick).
        """
        df = df.sort_values("Time").reset_index(drop=True).copy()
        df = compute_features(df)
        n = len(df)

        # transaction log: we will append a row for each tick
        transactions = []

        # simulator-like state for pnl
        position = None  # "LONG" or "SHORT" or None
        entry_price = None

        # buffers for PPO
        obs_list = []
        actions_list = []
        logp_list = []
        rewards_list = []
        values_list = []

        for t in range(n):
            price = df.at[t, 'Price']

            # Warmup: still record HOLD rows for each warmup tick
            if t < WARMUP - 1:
                obs = np.zeros(self.input_dim, dtype=np.float32)
                action = 0  # HOLD
                logp = math.log(1.0 / 4.0)
                value = 0.0
                reward = 0.0

                # append rollout storage
                obs_list.append(obs)
                actions_list.append(action)
                logp_list.append(logp)
                values_list.append(value)
                rewards_list.append(reward)

                # append transaction log (HOLD, profit 0)
                transactions.append({
                    "Time": df.at[t, "Time"],
                    "Price": price,
                    "Action": ACTION_MAP[action],
                    "Profit": 0.0
                })
                continue

            # Build observation vector
            dvol_log1p = df.at[t, 'dvol_log1p']
            ma60 = df.at[t, 'ma60']
            std60 = df.at[t, 'std60']
            rsi60 = df.at[t, 'rsi60']
            zscore = df.at[t, 'zscore60']
            pos_flag = 0.0
            entry_norm = 0.0
            if position == "LONG":
                pos_flag = 1.0
                entry_norm = (entry_price - price) / (price + 1e-8)
            elif position == "SHORT":
                pos_flag = -1.0
                entry_norm = (entry_price - price) / (price + 1e-8)

            obs = np.array([dvol_log1p, ma60, std60, rsi60, zscore, pos_flag, entry_norm], dtype=np.float32)

            # epsilon schedule
            if n > 1:
                eps = eps_end + (eps_start - eps_end) * (1 - t / (n - 1))
            else:
                eps = eps_end

            # select action
            action, logp, value = self.select_action(obs, epsilon=eps)

            # enforce constraints (no pyramiding, no repay if no position)
            if action == 1 and position == "LONG":
                action = 0
            if action == 2 and position == "SHORT":
                action = 0
            if action == 3 and position is None:
                action = 0

            action_str = ACTION_MAP[action]
            reward = 0.0
            realized = 0.0  # profit from repay (if any) this tick

            # Execute action
            if action_str == "BUY" and position is None:
                entry_price = price + SLIPPAGE
                position = "LONG"
                # record BUY (profit 0)
            elif action_str == "SELL" and position is None:
                entry_price = price - SLIPPAGE
                position = "SHORT"
            elif action_str == "REPAY" and position is not None:
                if position == "LONG":
                    exit_price = price - SLIPPAGE
                    realized = (exit_price - entry_price) * TRADE_UNIT
                else:  # SHORT
                    exit_price = price + SLIPPAGE
                    realized = (entry_price - exit_price) * TRADE_UNIT
                reward += realized
                # clear position
                position = None
                entry_price = None

            # Holding reward: 5% の一部を毎ティック付与
            if position is not None and entry_price is not None:
                if position == "LONG":
                    unreal = (price - entry_price) * TRADE_UNIT
                else:
                    unreal = (entry_price - price) * TRADE_UNIT
                reward += HOLD_REWARD_FACTOR * unreal

            # store rollout samples
            obs_list.append(obs)
            actions_list.append(action)
            logp_list.append(logp)
            values_list.append(value)
            rewards_list.append(reward)

            # RECORD TRANSACTION FOR THIS TICK (always one row per Time)
            transactions.append({
                "Time": df.at[t, "Time"],
                "Price": price,
                "Action": action_str,
                "Profit": float(realized)  # realized is non-zero only if REPAY
            })

        # End of day: if still holding position, force repay on last tick and adjust last transaction row
        if position is not None and n > 0:
            price = df.at[n - 1, 'Price']
            if position == "LONG":
                exit_price = price - SLIPPAGE
                realized = (exit_price - entry_price) * TRADE_UNIT
            else:
                exit_price = price + SLIPPAGE
                realized = (entry_price - exit_price) * TRADE_UNIT

            # Add forced repay as an additional final row (or modify last row if you prefer)
            transactions.append({
                "Time": df.at[n - 1, "Time"],
                "Price": price,
                "Action": "REPAY",
                "Profit": float(realized)
            })

            # Also add reward to last rollout entry so learning sees the final realized pnl
            if len(rewards_list) > 0:
                rewards_list[-1] += realized

            position = None
            entry_price = None

        # compute returns and advantages
        returns, advantages = self.compute_gae(rewards_list, values_list, gamma=GAMMA)

        # ppo update
        if len(obs_list) > 0:
            self.ppo_update(obs_list, actions_list, logp_list, returns, advantages)

        # save models
        torch.save(self.policy.state_dict(), self.policy_path)
        torch.save(self.value.state_dict(), self.value_path)

        df_trans = pd.DataFrame(transactions)
        # Ensure column order and types
        df_trans = df_trans[["Time", "Price", "Action", "Profit"]]
        return df_trans


# ---------------------------
# 小さなテスト / 実行例
# ---------------------------
if __name__ == "__main__":
    # ダミーデータで動作確認（実際はユーザーの1秒ティックデータを使う）
    times = np.arange(0, 3600, 1.0)  # 1時間分（例）
    # 簡易プライスシミュレーション
    prices = 10000 + np.cumsum(np.random.randn(len(times)) * 0.5)
    volumes = np.cumsum(np.random.poisson(10, size=len(times)))

    df = pd.DataFrame({"Time": times, "Price": prices, "Volume": volumes})
    trainer = Trainer(policy_path="policy.pth", value_path="value.pth")
    # 学習用（引け後）
    df_tx = trainer.train(df)
    print("Transactions after training:", df_tx)

    # 推論用（ザラ場）: 学習済みモデルが保存されているはず
    sim = TradingSimulator(model_path="policy.pth", value_path="value.pth", epsilon=0.0)
    for i in range(len(df)):
        action = sim.add(df.at[i, "Time"], df.at[i, "Price"], df.at[i, "Volume"])
        # 実運用では action を外部アプリに返す / ログする
        if i % 600 == 0:
            print(f"Tick {i} action: {action}")
