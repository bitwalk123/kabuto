# filename: ppo_tick_trader.py
# Requires: python==3.13.7, gymnasium==1.2.0, numpy==2.3.2, pandas==2.3.2, torch==2.8.0

import os
import math
import copy
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------
# Config / Hyperparameters
# ---------------------------
TICK_FEATURE_WINDOW = 60  # warm-up and feature window
TRADE_UNIT = 100
SLIPPAGE = 1.0  # in JPY per share (tick)
TICK_SIZE = 1.0  # assume 1 JPY tick
EPS_EXPLORE = 0.05  # epsilon for epsilon-greedy during training
UNREALIZED_COEF = 0.01  # fraction of unrealized P/L per tick as reward
PPO_CLIP = 0.2
PPO_EPOCHS = 4
PPO_LR = 3e-4
PPO_VALUE_COEF = 0.5
PPO_ENTROPY_COEF = 0.01
GAMMA = 0.99
GAE_LAMBDA = 0.95
MINIBATCH_SIZE = None  # using full-batch per epoch for simplicity (CPU)
MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = torch.device("cpu")


# ---------------------------
# Utilities for features
# ---------------------------
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts df with Time, Price, Volume (cumulative volume).
    Returns df with feature columns:
      - dlogvol = np.log1p(ΔVolume)
      - ma60, std60, rsi60, zscore60
    """
    df = df.copy().reset_index(drop=True)
    price = df["Price"].astype(float).to_numpy()
    vol = df["Volume"].astype(float).to_numpy()

    # ΔVolume: ensure non-negative (cumulative should be non-decreasing; but guard)
    dvol = np.zeros_like(vol)
    dvol[1:] = np.maximum(0.0, vol[1:] - vol[:-1])
    dlogvol = np.log1p(dvol)

    # rolling MA, STD
    ma = pd.Series(price).rolling(window=TICK_FEATURE_WINDOW, min_periods=1).mean().to_numpy()
    std = pd.Series(price).rolling(window=TICK_FEATURE_WINDOW, min_periods=1).std(ddof=0).fillna(0.0).to_numpy()

    # zscore = (price - ma)/std
    zscore = np.zeros_like(price)
    nonzero = std > 1e-8
    zscore[nonzero] = (price[nonzero] - ma[nonzero]) / std[nonzero]

    # RSI (60)
    # compute delta
    delta = np.zeros_like(price)
    delta[1:] = price[1:] - price[:-1]
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    # simple moving averages for gains/losses
    avg_gain = pd.Series(gain).rolling(window=TICK_FEATURE_WINDOW, min_periods=1).mean().to_numpy()
    avg_loss = pd.Series(loss).rolling(window=TICK_FEATURE_WINDOW, min_periods=1).mean().to_numpy()
    rs = np.zeros_like(price)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0.0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = np.nan_to_num(rsi, nan=50.0)  # neutral if undefined

    feat_df = df.copy()
    feat_df["dlogvol"] = dlogvol
    feat_df["ma60"] = ma
    feat_df["std60"] = std
    feat_df["zscore60"] = zscore
    feat_df["rsi60"] = rsi
    return feat_df


# ---------------------------
# Networks
# ---------------------------
class PolicyNet(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: List[int] = [64, 64], action_dim: int = 4):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.net(x)
        return logits  # raw logits for categorical distribution


class ValueNet(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: List[int] = [64, 64]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)  # shape (batch,)


# ---------------------------
# PPO Trainer and Simulator
# ---------------------------
ACTION_MAP = {
    0: "HOLD",
    1: "BUY",
    2: "SELL",
    3: "REPAY",
}


class TradingSimulator:
    """
    For PC2 (real-time inference). Loads models and on each `add` tick,
    returns an action string. Models must exist.
    """

    def __init__(self, model_dir: str = MODEL_DIR, feature_window: int = TICK_FEATURE_WINDOW):
        self.model_dir = model_dir
        self.feature_window = feature_window
        # load policy & value
        policy_path = os.path.join(model_dir, "policy.pth")
        value_path = os.path.join(model_dir, "value.pth")
        if not os.path.exists(policy_path) or not os.path.exists(value_path):
            raise FileNotFoundError(f"Required model files not found in {model_dir}")
        # model input dim computed from feature vector size = 5 (dlogvol, ma60, std60, zscore60, rsi60)
        self.input_dim = 5
        self.policy = PolicyNet(self.input_dim)
        self.value = ValueNet(self.input_dim)
        self.policy.load_state_dict(torch.load(policy_path, map_location=DEVICE))
        self.value.load_state_dict(torch.load(value_path, map_location=DEVICE))
        self.policy.eval()
        self.value.eval()

        # running buffer of raw df rows
        self._buffer = []
        self._feat_buffer = []
        # position state
        self.position = None  # None or dict {"side": "LONG" or "SHORT", "entry_price": float}

    def add(self, time: float, price: float, volume: float) -> str:
        """
        Called each tick by the user's app.
        Returns action string ("HOLD","BUY","SELL","REPAY") mapped from 0..3.
        """
        row = {"Time": time, "Price": float(price), "Volume": float(volume)}
        self._buffer.append(row)
        df = pd.DataFrame(self._buffer)
        feat_df = compute_features(df).iloc[-1]  # latest row features

        # If warm-up period not reached => HOLD
        if len(self._buffer) < self.feature_window:
            return "HOLD"

        # build feature vector
        x = np.array([
            feat_df["dlogvol"],
            feat_df["ma60"],
            feat_df["std60"],
            feat_df["zscore60"],
            feat_df["rsi60"],
        ], dtype=np.float32)
        x = torch.from_numpy(x).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = self.policy(x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy().flatten()
        action = int(np.argmax(probs))  # deterministic; trading app likely wants argmax
        # enforce position constraints: if position exists, disallow BUY/SELL (only REPAY/HOLD)
        if self.position is not None and action in [1, 2]:
            action = 0  # HOLD
        # If no position and action==REPAY -> treat as HOLD
        if self.position is None and action == 3:
            action = 0
        return ACTION_MAP[int(action)]


class Trainer:
    """
    Trainer for PC1 (after-hours). Accepts one-day df into train(df) and performs PPO updates.
    """

    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.input_dim = 5
        self.policy = PolicyNet(self.input_dim).to(DEVICE)
        self.value = ValueNet(self.input_dim).to(DEVICE)
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=PPO_LR)
        self.value_optim = optim.Adam(self.value.parameters(), lr=PPO_LR)

    def save_models(self):
        torch.save(self.policy.state_dict(), os.path.join(self.model_dir, "policy.pth"))
        torch.save(self.value.state_dict(), os.path.join(self.model_dir, "value.pth"))

    def load_models_if_exist(self):
        pfn = os.path.join(self.model_dir, "policy.pth")
        vfn = os.path.join(self.model_dir, "value.pth")
        if os.path.exists(pfn):
            self.policy.load_state_dict(torch.load(pfn, map_location=DEVICE))
        if os.path.exists(vfn):
            self.value.load_state_dict(torch.load(vfn, map_location=DEVICE))

    def select_action(self, x_np: np.ndarray, eps_explore: float = EPS_EXPLORE) -> Tuple[int, float]:
        """
        Return (action, log_prob) sampled from policy with epsilon-greedy exploration.
        """
        x = torch.from_numpy(x_np.astype(np.float32)).unsqueeze(0).to(DEVICE)
        logits = self.policy(x)
        probs = torch.softmax(logits, dim=-1).cpu().detach().numpy().flatten()
        if np.random.rand() < eps_explore:
            action = np.random.choice(len(probs))
        else:
            action = int(np.random.choice(len(probs), p=probs))
        # compute log_prob
        logp = torch.log_softmax(logits, dim=-1)[0, action].item()
        return action, logp

    def compute_gae(self, rewards, values, dones, gamma=GAMMA, lam=GAE_LAMBDA):
        """
        rewards, values: lists/np arr
        returns advantages, returns
        """
        adv = np.zeros_like(rewards, dtype=np.float32)
        lastgaelam = 0.0
        # append value at end for bootstrap
        values = np.append(values, 0.0)
        for t in reversed(range(len(rewards))):
            nonterminal = 1.0 - float(dones[t])
            delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
            adv[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        returns = adv + values[:-1]
        return adv, returns

    def ppo_update(self, obs, actions, old_log_probs, returns, advantages):
        """
        obs: np.array (N, D)
        actions: np.array (N,)
        old_log_probs: np.array (N,)
        returns, advantages: np.array (N,)
        """
        obs_tensor = torch.from_numpy(obs.astype(np.float32)).to(DEVICE)
        actions_tensor = torch.from_numpy(actions).long().to(DEVICE)
        old_logp_tensor = torch.from_numpy(old_log_probs.astype(np.float32)).to(DEVICE)
        returns_tensor = torch.from_numpy(returns.astype(np.float32)).to(DEVICE)
        adv_tensor = torch.from_numpy(advantages.astype(np.float32)).to(DEVICE)

        for _ in range(PPO_EPOCHS):
            # Forward
            logits = self.policy(obs_tensor)
            logp_all = torch.log_softmax(logits, dim=-1)
            logp = logp_all.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
            ratio = torch.exp(logp - old_logp_tensor)

            # policy loss
            surr1 = ratio * adv_tensor
            surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * adv_tensor
            policy_loss = -torch.min(surr1, surr2).mean()

            # entropy
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(axis=1).mean()

            # value loss
            values = self.value(obs_tensor)
            value_loss = nn.functional.mse_loss(values, returns_tensor)

            # combined
            loss = policy_loss + PPO_VALUE_COEF * value_loss - PPO_ENTROPY_COEF * entropy

            # step
            self.policy_optim.zero_grad()
            self.value_optim.zero_grad()
            loss.backward()
            # gradient clipping for stability
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            nn.utils.clip_grad_norm_(self.value.parameters(), 0.5)
            self.policy_optim.step()
            self.value_optim.step()

    def train(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        df: Time, Price, Volume (cumulative)
        returns df_transaction with Time, Price, Volume, Action, Profit
        """

        # load existing models (so training can continue from previous)
        self.load_models_if_exist()

        # prepare features
        feat_df = compute_features(df)
        n = len(feat_df)
        # storage for simulation
        df_transaction = pd.DataFrame(columns=["Time", "Price", "Volume", "Action", "Profit"])
        df_transaction = df_transaction.astype(object)

        # trajectory buffers for PPO
        obs_buf = []
        act_buf = []
        logp_buf = []
        rew_buf = []
        val_buf = []
        done_buf = []

        # position state
        position = None  # None or dict {"side": "LONG"/"SHORT", "entry_price": float}
        pending_profit = 0.0  # profit realized at repay

        for t in range(n):
            row = feat_df.iloc[t]
            time = row["Time"]
            price = float(row["Price"])
            volume = float(row["Volume"])

            # warm-up period
            if t < TICK_FEATURE_WINDOW:
                action_idx = 0  # HOLD
                logp = 0.0
                value_est = 0.0
                reward = 0.0
                done = False
                # record
                df_transaction.at[t, "Time"] = time
                df_transaction.at[t, "Price"] = price
                df_transaction.at[t, "Volume"] = volume
                df_transaction.at[t, "Action"] = "HOLD"
                df_transaction.at[t, "Profit"] = 0.0
                # For PPO buffers, we can still push a zero transition to keep lengths consistent
                obs_buf.append(np.zeros(self.input_dim, dtype=np.float32))
                act_buf.append(action_idx)
                logp_buf.append(logp)
                rew_buf.append(0.0)
                val_buf.append(value_est)
                done_buf.append(done)
                continue

            # build observation vector: [dlogvol, ma60, std60, zscore60, rsi60]
            obs = np.array([
                row["dlogvol"],
                row["ma60"],
                row["std60"],
                row["zscore60"],
                row["rsi60"],
            ], dtype=np.float32)

            # select action via policy (with epsilon-greedy)
            action_idx, logp = self.select_action(obs, eps_explore=EPS_EXPLORE)

            # enforce trade constraints:
            if position is not None and action_idx in [1, 2]:
                # cannot open new if already holding; force HOLD
                action_idx = 0
            if position is None and action_idx == 3:
                # cannot repay if no position
                action_idx = 0

            # compute immediate reward:
            reward = 0.0
            realized_profit = 0.0

            # value estimate for baseline
            with torch.no_grad():
                val_tensor = self.value(torch.from_numpy(obs).unsqueeze(0))
                value_est = float(val_tensor.item())

            # Action semantics:
            # 1 BUY (open LONG): entry = price + slippage
            # 2 SELL (open SHORT): entry = price - slippage
            # 3 REPAY: if LONG -> exit = price - slippage (profit = (exit - entry) * TRADE_UNIT)
            #          if SHORT -> exit = price + slippage (profit = (entry - exit) * TRADE_UNIT)
            if action_idx == 1 and position is None:
                entry_price = price + SLIPPAGE
                position = {"side": "LONG", "entry_price": float(entry_price)}
                # no immediate reward
            elif action_idx == 2 and position is None:
                entry_price = price - SLIPPAGE
                position = {"side": "SHORT", "entry_price": float(entry_price)}
            elif action_idx == 3 and position is not None:
                # repay: calculate realized profit
                if position["side"] == "LONG":
                    exit_price = price - SLIPPAGE
                    profit = (exit_price - position["entry_price"]) * TRADE_UNIT
                else:  # SHORT
                    exit_price = price + SLIPPAGE
                    profit = (position["entry_price"] - exit_price) * TRADE_UNIT
                realized_profit = float(profit)
                reward += realized_profit  # add realized profit as reward
                position = None
            else:
                # HOLD or other no-op
                pass

            # while holding, give a small per-tick unrealized P/L reward (to encourage good holding)
            if position is not None:
                if position["side"] == "LONG":
                    unrealized = (price - position["entry_price"]) * TRADE_UNIT
                else:
                    unrealized = (position["entry_price"] - price) * TRADE_UNIT
                reward += UNREALIZED_COEF * float(unrealized)

            # record step
            df_transaction.at[t, "Time"] = time
            df_transaction.at[t, "Price"] = price
            df_transaction.at[t, "Volume"] = volume
            df_transaction.at[t, "Action"] = ACTION_MAP[int(action_idx)]
            df_transaction.at[t, "Profit"] = realized_profit if realized_profit != 0.0 else 0.0

            # done flag is False for per-tick (we'll set last tick done)
            done = False

            # append to buffers
            obs_buf.append(obs)
            act_buf.append(int(action_idx))
            logp_buf.append(float(logp))
            rew_buf.append(float(reward))
            val_buf.append(float(value_est))
            done_buf.append(bool(done))

        # End of day: if position remains, force repay at last price
        if position is not None:
            last_price = float(df.iloc[-1]["Price"])
            if position["side"] == "LONG":
                exit_price = last_price - SLIPPAGE
                profit = (exit_price - position["entry_price"]) * TRADE_UNIT
            else:
                exit_price = last_price + SLIPPAGE
                profit = (position["entry_price"] - exit_price) * TRADE_UNIT
            # append final repay as last time step (we'll put at last index)
            t_last = n - 1
            df_transaction.at[t_last, "Action"] = "REPAY"
            df_transaction.at[t_last, "Profit"] = float(profit)
            # also add reward to last transition
            if len(rew_buf) >= 1:
                rew_buf[-1] += float(profit)
            position = None

        # mark the last as done
        if len(done_buf) > 0:
            done_buf[-1] = True

        # Convert buffers to numpy
        obs_arr = np.vstack([o for o in obs_buf]).astype(np.float32)
        acts = np.array(act_buf, dtype=np.int64)
        old_logps = np.array(logp_buf, dtype=np.float32)
        rewards = np.array(rew_buf, dtype=np.float32)
        vals = np.array(val_buf, dtype=np.float32)
        dones = np.array(done_buf, dtype=np.float32)

        # compute advantages and returns
        advantages, returns = self.compute_gae(rewards, vals, dones, gamma=GAMMA, lam=GAE_LAMBDA)
        # normalize advantages
        adv_mean = advantages.mean() if advantages.size > 0 else 0.0
        adv_std = advantages.std() if advantages.size > 0 else 1.0
        if adv_std < 1e-8:
            adv_std = 1.0
        advantages = (advantages - adv_mean) / adv_std

        # PPO update
        if len(obs_arr) > 0:
            self.ppo_update(obs_arr, acts, old_logps, returns, advantages)

        # save models
        self.save_models()

        # ensure df_transaction columns types consistent
        df_transaction = df_transaction.fillna(0.0).reset_index(drop=True)
        # Guarantee that every Time has an Action string and Profit numeric
        # (already ensured above)
        return df_transaction


# ---------------------------
# Example usage (training script)
# ---------------------------
if __name__ == "__main__":
    # This block demonstrates how to call Trainer.train() with a day's DataFrame.
    # The user supplies their own file reading function get_excel_sheet as in the spec.
    # Here is a simple synthetic example:
    import datetime

    # Build synthetic one-day tick data for demonstration (replace with actual get_excel_sheet)
    n_ticks = 2000  # real use-case: ~19,500
    times = np.arange(n_ticks)  # seconds from market open (example)
    # synthetic price walk
    prices = 1000 + np.cumsum(np.random.randn(n_ticks) * 0.5)
    # cumulative volume
    dvol = np.random.poisson(10, size=n_ticks)
    vols = np.cumsum(dvol)

    df_day = pd.DataFrame({
        "Time": times,
        "Price": prices,
        "Volume": vols,
    })

    trainer = Trainer()
    df_trades = trainer.train(df_day)
    print("Sample of trade transaction log:")
    print(df_trades.head(20))
    # models saved in MODEL_DIR
