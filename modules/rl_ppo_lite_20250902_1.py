# filename: ppo_trading_day.py
# Python: 3.13.7
# Dependences: gymnasium==1.2.0, numpy==2.3.2, pandas==2.3.2, torch==2.8.0

import os
from typing import Tuple, List, Dict, Optional
import copy
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# 固定設定（要件）
TRADE_UNIT = 100
SLIPPAGE = 1  # 1 tick
WARMUP = 60
TICK_PER_DAY_APPROX = 19500

DEVICE = torch.device("cpu")  # CPU 前提


# ---------------------------
# ユーティリティ（特徴量計算）
# ---------------------------
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    df: Time, Price, Volume (累計)
    返り値: df に特徴量カラムを追加して返す
    特徴:
      - dvol = diff累計 -> ΔVolume -> np.log1p
      - MA60, STD60, RSI60, Zscore60
    """
    df = df.copy().reset_index(drop=True)
    # ΔVolume: 累計 -> 差分。先頭は 0
    dvol = df["Volume"].diff().fillna(0).clip(lower=0).astype(float)  # 負の増分は 0 とする（保守的）
    df["dvol_log1p"] = np.log1p(dvol)

    # Rolling stats on Price
    n = 60
    df["MA60"] = df["Price"].rolling(n, min_periods=1).mean()
    df["STD60"] = df["Price"].rolling(n, min_periods=1).std(ddof=0).fillna(0.0)

    # RSI: using typical RSI on price changes (gain/loss)
    delta = df["Price"].diff().fillna(0.0)
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    # Simple RSI with n period average
    avg_up = up.rolling(n, min_periods=1).mean()
    avg_down = down.rolling(n, min_periods=1).mean()
    rs = avg_up / (avg_down + 1e-8)
    df["RSI60"] = 100 - (100 / (1 + rs))

    # Z-score over window n: (price - mean)/std
    df["Z60"] = (df["Price"] - df["MA60"]) / (df["STD60"] + 1e-8)

    # Fill NaN
    df.fillna(method="bfill", inplace=True)
    df.fillna(0.0, inplace=True)
    return df


# ---------------------------
# 環境（シミュレータ）: 内部で売買制約を強制
# ---------------------------
class TradingEnv:
    """
    内部状態:
      - position: 0 (flat), +1 (long), -1 (short)
      - entry_price: price at which position opened (per-share)
    Action mapping:
      0: HOLD
      1: BUY (open long)    -> entry = price + slippage
      2: SELL (open short)  -> entry = price - slippage
      3: REPAY (close)      -> exit logic depends on position
    制約:
      - position != 0 のとき BUY/SELL は禁止（代わりに HOLD に変換）
      - warmup の間は必ず HOLD（上位呼び出しで制御可）
    """

    def __init__(self, trade_unit=TRADE_UNIT, slippage=SLIPPAGE):
        self.trade_unit = trade_unit
        self.slippage = slippage
        self.position = 0
        self.entry_price = 0.0
        self.realized_pnl = 0.0
        self.unrealized = 0.0

    def reset(self):
        self.position = 0
        self.entry_price = 0.0
        self.realized_pnl = 0.0
        self.unrealized = 0.0

    def step(self, action: int, price: float) -> Tuple[float, float]:
        """
        action: 0..3
        price: current market price (ticker price)
        returns: (reward, realized_profit_if_any)
        """
        reward = 0.0
        realized = 0.0

        # Enforce action constraints:
        if self.position != 0 and action in (1, 2):
            # cannot open a new opposite/other position while holding; force HOLD
            action = 0

        # Execute
        if action == 1:  # BUY: open long
            if self.position == 0:
                entry = price + self.slippage
                self.position = +1
                self.entry_price = entry
                # no immediate reward
        elif action == 2:  # SELL: open short
            if self.position == 0:
                entry = price - self.slippage
                self.position = -1
                self.entry_price = entry
        elif action == 3:  # REPAY: close existing
            if self.position == +1:
                exit_price = price - self.slippage
                profit = (exit_price - self.entry_price) * self.trade_unit
                realized = profit
                self.realized_pnl += profit
                reward += profit  # on repay reward includes realized profit
                self.position = 0
                self.entry_price = 0.0
            elif self.position == -1:
                exit_price = price + self.slippage
                profit = (self.entry_price - exit_price) * self.trade_unit
                realized = profit
                self.realized_pnl += profit
                reward += profit
                self.position = 0
                self.entry_price = 0.0
        # HOLD or invalid action does nothing

        # Update unrealized
        if self.position == +1:
            self.unrealized = (price - self.entry_price) * self.trade_unit
        elif self.position == -1:
            self.unrealized = (self.entry_price - price) * self.trade_unit
        else:
            self.unrealized = 0.0

        return reward, realized


# ---------------------------
# ネットワーク（方策・価値 別々）
# ---------------------------
class PolicyNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, n_actions: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x):
        return self.net(x)  # logits


class ValueNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------
# PPO エージェント（シンプル）
# ---------------------------
class PPOAgent:
    def __init__(
        self,
        input_dim: int,
        n_actions: int = 4,
        policy_lr: float = 3e-4,
        value_lr: float = 1e-3,
        clip_epsilon: float = 0.2,
        epochs: int = 4,
        gamma: float = 0.99,
        lam: float = 0.95,
        entropy_coef: float = 1e-3,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
    ):
        self.policy = PolicyNet(input_dim, hidden_dim=128, n_actions=n_actions).to(DEVICE)
        self.value = ValueNet(input_dim, hidden_dim=128).to(DEVICE)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=value_lr)
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.gamma = gamma
        self.lam = lam
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

    def get_action_and_logp(self, obs: torch.Tensor) -> Tuple[int, float, torch.Tensor]:
        logits = self.policy(obs)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        logp = dist.log_prob(action)
        return int(action.item()), float(logp.item()), probs.detach()

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        logits = self.policy(obs)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        logp = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        values = self.value(obs)
        return logp, values, entropy

    def compute_gae(self, rewards, values, dones):
        """
        rewards: list
        values: list (len = T+1) with bootstrap last
        dones: list of bools
        returns: advantages, returns (both numpy arrays)
        """
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        lastgaelam = 0
        for t in reversed(range(T)):
            nonterminal = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * values[t + 1] * nonterminal - values[t]
            advantages[t] = lastgaelam = delta + self.gamma * self.lam * nonterminal * lastgaelam
        returns = advantages + values[:-1]
        return advantages, returns

    def update(self, batch_obs, batch_actions, batch_logp_old, batch_returns, batch_adv):
        obs = torch.tensor(batch_obs, dtype=torch.float32, device=DEVICE)
        actions = torch.tensor(batch_actions, dtype=torch.long, device=DEVICE)
        old_logp = torch.tensor(batch_logp_old, dtype=torch.float32, device=DEVICE)
        returns = torch.tensor(batch_returns, dtype=torch.float32, device=DEVICE)
        adv = torch.tensor(batch_adv, dtype=torch.float32, device=DEVICE)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for _ in range(self.epochs):
            logp, values, entropy = self.evaluate_actions(obs, actions)
            ratio = torch.exp(logp - old_logp)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * adv
            policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
            value_loss = (returns - values).pow(2).mean() * self.value_coef

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy_optimizer.step()

            self.value_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
            self.value_optimizer.step()

    def save(self, path: str):
        torch.save(
            {
                "policy_state": self.policy.state_dict(),
                "value_state": self.value.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=DEVICE)
        self.policy.load_state_dict(checkpoint["policy_state"])
        self.value.load_state_dict(checkpoint["value_state"])


# ---------------------------
# Trainer クラス（学習用：PC1）
# ---------------------------
class Trainer:
    """
    使用法:
      trainer = Trainer()
      df_transactions = trainer.train(df_day)  # df_day has Time, Price, Volume (累計)
      trainer.save_model("policy.pth")
    """

    def __init__(
        self,
        input_features: List[str] = ["Price", "dvol_log1p", "MA60", "STD60", "RSI60", "Z60"],
        model_path: str = "policy.pth",
        epochs: int = 20,
        batch_size_steps: int = 4096,
        epsilon_greedy: float = 0.05,
    ):
        self.input_features = input_features
        self.model_path = model_path
        self.epochs = epochs
        self.batch_size_steps = batch_size_steps
        self.epsilon_greedy = epsilon_greedy  # for exploration
        self.agent = PPOAgent(input_dim=len(self.input_features))
        self.env = TradingEnv()
        # random seeds for reproducibility (optional)
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

    def observation_from_row(self, row: pd.Series) -> np.ndarray:
        # Observation vector: choose normalized features (simple)
        obs = np.array([row[f] for f in self.input_features], dtype=np.float32)
        # Simple scaling for Price to avoid large magnitude: divide by 1e3 (assumption)
        obs[0] = obs[0] / 1000.0
        return obs

    def action_by_policy(self, obs_np: np.ndarray, greedy: bool = False) -> Tuple[int, float]:
        obs = torch.tensor(obs_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        logits = self.agent.policy(obs)
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy().squeeze(0)
        if greedy:
            action = int(np.argmax(probs))
            logp = float(math.log(max(probs[action], 1e-12)))
            return action, logp
        # ε-greedy: with prob epsilon choose random action (to encourage exploration)
        if random.random() < self.epsilon_greedy:
            action = random.randint(0, 3)
            logp = math.log(1.0 / 4.0)
            return action, logp
        # sample from categorical
        action = int(np.random.choice(len(probs), p=probs))
        logp = float(math.log(max(probs[action], 1e-12)))
        return action, logp

    def train(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        df: single day tick data (Time, Price, Volume cumulative)
        returns: df_transaction with Time, Price, Action (string), Profit (realized at repay else 0)
        """
        df = compute_features(df)
        n_rows = len(df)
        df_transactions = pd.DataFrame(
            {
                "Time": df["Time"].values,
                "Price": df["Price"].values,
                "Action": ["HOLD"] * n_rows,
                "Profit": [0.0] * n_rows,
            }
        )

        for epoch in range(self.epochs):
            # Reset env & buffers
            self.env.reset()
            obs_buf = []
            act_buf = []
            logp_buf = []
            rew_buf = []
            val_buf = []
            dones = []

            # Walk through day sequentially (on-policy)
            for t in range(n_rows):
                row = df.iloc[t]
                # warmup constraint
                if t < WARMUP:
                    action = 0  # HOLD
                    logp = math.log(1.0)
                else:
                    obs_np = self.observation_from_row(row)
                    action, logp = self.action_by_policy(obs_np)
                    # enforce env-level constraint: if action invalid, environment will override to HOLD internally by step
                # value estimate
                with torch.no_grad():
                    val = float(self.agent.value(torch.tensor(obs_np if t>=WARMUP else np.zeros(len(self.input_features)), dtype=torch.float32).unsqueeze(0)))
                # env.step
                price = float(row["Price"])
                reward, realized = self.env.step(action, price)
                # mid-hold reward (含み益の一部 per-tick)
                if self.env.position != 0:
                    # give a small fraction (e.g., 1%) of unrealized as shaping reward each tick
                    shaping = 0.01 * self.env.unrealized
                    reward += shaping
                # save transaction record for this timestep if action changed state
                action_str = ["HOLD", "BUY", "SELL", "REPAY"][action]
                df_transactions.at[t, "Action"] = action_str
                df_transactions.at[t, "Profit"] = realized

                # append to buffers
                obs_buf.append(self.observation_from_row(row) if t >= WARMUP else np.zeros(len(self.input_features), dtype=np.float32))
                act_buf.append(action)
                logp_buf.append(logp)
                rew_buf.append(reward)
                val_buf.append(val)
                dones.append(False)

            # force repay at end if position exists
            if self.env.position != 0:
                # close at last price
                last_price = float(df.iloc[-1]["Price"])
                reward, realized = self.env.step(3, last_price)
                # record as final repay in transaction table (append row)
                df_transactions.at[n_rows - 1, "Action"] = "REPAY"
                df_transactions.at[n_rows - 1, "Profit"] += realized
                # append reward to buffers (as final step)
                rew_buf[-1] += reward  # add to last step

            # bootstrap value for last state
            with torch.no_grad():
                last_val = 0.0
            values = np.array(val_buf + [last_val], dtype=np.float32)

            # compute advantages & returns
            advantages, returns = self.agent.compute_gae(rew_buf, values, dones)

            # update PPO
            self.agent.update(
                batch_obs=np.array(obs_buf, dtype=np.float32),
                batch_actions=np.array(act_buf, dtype=np.int32),
                batch_logp_old=np.array(logp_buf, dtype=np.float32),
                batch_returns=returns,
                batch_adv=advantages,
            )

            # simple logging
            epoch_realized = self.env.realized_pnl
            print(f"[Epoch {epoch+1}/{self.epochs}] realized pnl this epoch: {epoch_realized:.0f}")

        # training done: save model
        self.save_model(self.model_path)
        print(f"Model saved to {self.model_path}")

        # Ensure all rows have Action text and Profit as float
        df_transactions["Profit"] = df_transactions["Profit"].astype(float)
        return df_transactions

    def save_model(self, path: Optional[str] = None):
        p = path or self.model_path
        self.agent.save(p)

    def load_model(self, path: Optional[str] = None):
        p = path or self.model_path
        self.agent.load(p)


# ---------------------------
# TradingSimulator（推論用：PC2）
# ---------------------------
class TradingSimulator:
    """
    PC2 の推論専用クラス
    Usage:
      sim = TradingSimulator(model_path="policy.pth")
      sim.add(time, price, volume)  # called every tick (approx 1s)
      -> returns action string among ["HOLD","BUY","SELL","REPAY"]
    - 内部でモデルが無ければ例外を投げる。
    - 内部は同じ特徴量計算ロジックを使用。60ティックのウォームアップを守る。
    - 日中のティックを add で受け取り、その日のティックを内部に記録（必要なら later save）
    """

    def __init__(self, model_path: str = "policy.pth", input_features: List[str] = ["Price", "dvol_log1p", "MA60", "STD60", "RSI60", "Z60"], epsilon_greedy: float = 0.0):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        self.model_path = model_path
        self.input_features = input_features
        self.epsilon_greedy = epsilon_greedy
        self.env = TradingEnv()
        self._buffer = []  # store dicts of Time, Price, Volume (cumulative)
        self._model = PPOAgent(input_dim=len(self.input_features))
        self._model.load(self.model_path)
        self._warmup = WARMUP

        # For online rolling features, keep small DF
        self._df = pd.DataFrame(columns=["Time", "Price", "Volume"])

    def _compute_latest_features(self) -> pd.Series:
        df = self._df.copy().reset_index(drop=True)
        df = compute_features(df)
        return df.iloc[-1]

    def add(self, time: float, price: float, volume: float) -> str:
        """
        Called per tick.
        Returns action string: "HOLD","BUY","SELL","REPAY"
        """
        # append to df
        self._df.loc[len(self._df)] = {"Time": time, "Price": price, "Volume": volume}

        t = len(self._df) - 1
        if t < self._warmup:
            return "HOLD"
        row = self._compute_latest_features()

        # build observation
        obs = np.array([row[f] for f in self.input_features], dtype=np.float32)
        obs[0] = obs[0] / 1000.0  # same scaling as trainer

        # policy forward
        with torch.no_grad():
            logits = self._model.policy(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
            probs = torch.softmax(logits, dim=-1).cpu().numpy().squeeze(0)
        # epsilon_greedy: optional small randomization
        if random.random() < self.epsilon_greedy:
            action = random.randint(0, 3)
        else:
            action = int(np.argmax(probs))

        # Enforce action constraints: if trying to open while position exists, force HOLD
        if self.env.position != 0 and action in (1, 2):
            action = 0

        # Execute on env
        reward, realized = self.env.step(action, price)

        action_str = ["HOLD", "BUY", "SELL", "REPAY"][action]
        return action_str

    def save_record(self, path: str):
        self._df.to_csv(path, index=False)


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    # Example: prepare synthetic df or load real day df with Time, Price, Volume(cumulative)
    # Here we create a small synthetic sample for quick test
    times = np.arange(0, 500)  # 500 ticks
    prices = 1000 + np.cumsum(np.random.randn(len(times)) * 2.0)  # random walk
    volumes = np.cumsum(np.random.randint(1, 100, size=len(times)))
    df_day = pd.DataFrame({"Time": times, "Price": prices, "Volume": volumes})

    trainer = Trainer(model_path="policy.pth", epochs=3)  # epochs small for quick test
    df_transactions = trainer.train(df_day)
    print(df_transactions.head(20))

    # Simulate online (PC2)
    sim = TradingSimulator(model_path="policy.pth", epsilon_greedy=0.0)
    for i in range(len(df_day)):
        a = sim.add(float(df_day.loc[i, "Time"]), float(df_day.loc[i, "Price"]), float(df_day.loc[i, "Volume"]))
        if i % 100 == 0:
            print(f"tick {i} -> {a}")
    sim.save_record("today_ticks.csv")
