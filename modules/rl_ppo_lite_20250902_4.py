# filename: rl_trading_sample.py
# Requires: python==3.13.7, gymnasium==1.2.0, numpy==2.3.2, pandas==2.3.2, torch==2.8.0

import os
import math
import random
import typing as tp
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------
# Utility / Config
# ---------------------------
SEED = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
DEVICE = torch.device("cpu")

ACTION_MAP = {0: "HOLD", 1: "BUY", 2: "SELL", 3: "REPAY"}
INV_ACTION = {v: k for k, v in ACTION_MAP.items()}

TRADE_UNIT = 100
SLIPPAGE = 1  # ticks (JPY)
WARMUP = 60  # first N ticks MUST HOLD

# ---------------------------
# Networks
# ---------------------------
class PolicyNet(nn.Module):
    def __init__(self, obs_dim, hidden=128, n_actions=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)  # raw logits


class ValueNet(nn.Module):
    def __init__(self, obs_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------
# Feature engineering
# ---------------------------
def compute_features(df: pd.DataFrame) -> np.ndarray:
    # df must have Time, Price, Volume (cum volume)
    price = df["Price"].to_numpy(dtype=float)
    vol = df["Volume"].to_numpy(dtype=float)

    # ΔVolume: difference of cumulative -> incremental
    dvol = np.zeros_like(vol)
    dvol[1:] = vol[1:] - vol[:-1]
    dvol = np.maximum(dvol, 0.0)
    log_dvol = np.log1p(dvol)  # shape (T,)

    # rolling stats with window W
    W = WARMUP
    ma = pd.Series(price).rolling(W, min_periods=1).mean().to_numpy()
    std = pd.Series(price).rolling(W, min_periods=1).std(ddof=0).fillna(0).to_numpy()
    # RSI (n=W)
    delta = pd.Series(price).diff().fillna(0)
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = up.rolling(W, min_periods=1).mean()
    avg_down = down.rolling(W, min_periods=1).mean()
    rs = avg_up / (avg_down + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.to_numpy()

    # Z-score
    z = np.zeros_like(price)
    # avoid divide by zero
    z = (price - ma) / (std + 1e-8)

    # Stack features: Price (normalized), log_dvol, ma, std, rsi, z
    # normalize price by ma to reduce scale differences
    price_norm = price / (ma + 1e-8)
    feats = np.vstack([price_norm, log_dvol, ma, std, rsi, z]).T  # shape (T, feat_dim)
    return feats.astype(np.float32)


# ---------------------------
# Trading environment logic (used by Trainer training loop)
# ---------------------------
@dataclass
class Position:
    side: str = "NONE"  # "LONG" or "SHORT" or "NONE"
    entry_price: float = 0.0  # entry price (per-share)
    size: int = 0  # number of shares (here either 0 or TRADE_UNIT)


class TradingEnv:
    def __init__(self, prices: np.ndarray):
        self.position = Position()
        self.prices = prices

    def can_open(self):
        return self.position.size == 0

    def open_long(self, price):
        # entry price includes slippage
        entry = price + SLIPPAGE
        self.position = Position(side="LONG", entry_price=entry, size=TRADE_UNIT)
        return entry

    def open_short(self, price):
        entry = price - SLIPPAGE
        self.position = Position(side="SHORT", entry_price=entry, size=TRADE_UNIT)
        return entry

    def repay(self, price):
        # calculates realized profit for current position, then reset
        if self.position.size == 0:
            return 0.0
        if self.position.side == "LONG":
            exit_price = price - SLIPPAGE
            profit_per_share = exit_price - self.position.entry_price
        else:  # SHORT
            exit_price = price + SLIPPAGE
            profit_per_share = self.position.entry_price - exit_price
        realized = profit_per_share * self.position.size
        self.position = Position()  # reset
        return float(realized)

    def unrealized(self, price):
        if self.position.size == 0:
            return 0.0
        if self.position.side == "LONG":
            current = (price - SLIPPAGE) - self.position.entry_price
        else:
            current = self.position.entry_price - (price + SLIPPAGE)
        return float(current * self.position.size)


# ---------------------------
# PPO helper functions
# ---------------------------
def compute_gae(rewards, masks, values, gamma=0.99, lam=0.95):
    # rewards: list of floats; masks: 1 if not terminal, else 0
    values = np.append(values, 0.0)
    gae = 0.0
    returns = np.zeros_like(rewards)
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * lam * masks[step] * gae
        returns[step] = gae + values[step]
    adv = returns - values[:-1]
    return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-8)


# ---------------------------
# Trainer class
# ---------------------------
class Trainer:
    def __init__(
        self,
        policy_path: str = "policy.pth",
        value_path: str = "value.pth",
        ppo_epochs: int = 8,
        batch_size: int = 256,
        lr: float = 3e-4,
        epsilon: float = 0.2,
        entropy_coef: float = 1e-3,
        value_coef: float = 0.5,
        exploration_eps: float = 0.1,
    ):
        self.policy_path = policy_path
        self.value_path = value_path
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.epsilon = epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.exploration_eps = exploration_eps

        # models will be lazily created on train
        self.policy = None
        self.value = None
        self.policy_optim = None
        self.value_optim = None

    def save_models(self):
        torch.save(self.policy.state_dict(), self.policy_path)
        torch.save(self.value.state_dict(), self.value_path)

    def load_models(self):
        self.policy = PolicyNet(self.obs_dim)
        self.value = ValueNet(self.obs_dim)
        self.policy.load_state_dict(torch.load(self.policy_path, map_location=DEVICE))
        self.value.load_state_dict(torch.load(self.value_path, map_location=DEVICE))
        self.policy.to(DEVICE)
        self.value.to(DEVICE)

    def _init_models(self, obs_dim):
        self.obs_dim = obs_dim
        self.policy = PolicyNet(obs_dim)
        self.value = ValueNet(obs_dim)
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.value_optim = optim.Adam(self.value.parameters(), lr=self.lr)

    def select_action(self, obs: np.ndarray, hidden_state=None, deterministic=False):
        # obs: (obs_dim,)
        x = torch.from_numpy(obs).float().unsqueeze(0).to(DEVICE)
        logits = self.policy(x)
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]
        # epsilon-greedy exploration (sample uniformly with prob eps)
        if (not deterministic) and (random.random() < self.exploration_eps):
            action = random.randrange(len(probs))
            logp = math.log(probs[action] + 1e-8)
        else:
            action = int(np.argmax(probs))
            logp = math.log(probs[action] + 1e-8)
        return action, logp, probs

    def train(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Train PPO on a single day's tick data and return df_transaction with Time, Price, Action, Profit.
        """
        df = df.reset_index(drop=True).copy()
        assert {"Time", "Price", "Volume"}.issubset(df.columns), "df must have Time, Price, Volume columns"

        feats = compute_features(df)
        T, obs_dim = feats.shape
        self._init_models(obs_dim)

        # Convert to tensors once
        prices = df["Price"].to_numpy(dtype=float)

        # We'll collect transitions over the single pass (one epoch over T ticks), then perform PPO updates.
        # To create more data we can repeat multiple "episodes" where policy acts over the day's ticks.
        trajectories = []

        # Hyper: number of episodes (passes over the day's ticks) per train() call
        n_episodes = 8

        for ep in range(n_episodes):
            env = TradingEnv(prices)
            obs_buf = []
            act_buf = []
            logp_buf = []
            rew_buf = []
            val_buf = []
            mask_buf = []
            profit_buf = []  # store realized profit at each tick (0 or realized)
            action_names = []

            for t in range(T):
                obs = feats[t]
                # respect warmup: force HOLD in warmup
                if t < WARMUP:
                    action = 0  # HOLD
                    logp = math.log(1.0)  # dummy
                    probs = np.ones(4) / 4.0
                else:
                    action, logp, probs = self.select_action(obs)

                    # enforce constraint: if position open, disallow opening opposite side
                    if env.position.size > 0:
                        # allow only HOLD or REPAY
                        if action in (1, 2):  # BUY or SELL (open)
                            action = 0  # force HOLD
                    else:
                        # if no position, disallow REPAY
                        if action == 3:
                            action = 0

                # Compute immediate reward and env step
                price = prices[t]
                realized = 0.0
                # default reward per tick: small part of unrealized P/L if holding
                if action == 1:  # BUY
                    if env.can_open():
                        env.open_long(price)
                elif action == 2:  # SELL (open short)
                    if env.can_open():
                        env.open_short(price)
                elif action == 3:  # REPAY
                    realized = env.repay(price)
                # per-tick unrealized reward
                unreal = env.unrealized(price)
                # give a small fraction of unrealized as intermediate reward to encourage momentum holding
                tick_reward = 0.01 * unreal
                # when repay, add full realized profit as well
                total_reward = tick_reward + realized

                obs_buf.append(obs)
                act_buf.append(action)
                logp_buf.append(logp)
                rew_buf.append(total_reward)
                val_est = self.value(torch.from_numpy(obs).float().unsqueeze(0)).detach().cpu().numpy()[0]
                val_buf.append(float(val_est))
                mask_buf.append(1.0)  # no terminal in this simple formulation
                profit_buf.append(realized)
                action_names.append(ACTION_MAP[action])

            # Force repay at end if position open
            if env.position.size > 0:
                realized = env.repay(prices[-1])
                rew_buf[-1] += realized
                profit_buf[-1] += realized
                action_names[-1] = ACTION_MAP[3]  # mark as REPAY in last tick

            # Save trajectory
            trajectories.append({
                "obs": np.array(obs_buf),
                "acts": np.array(act_buf),
                "logps": np.array(logp_buf),
                "rews": np.array(rew_buf),
                "vals": np.array(val_buf),
                "masks": np.array(mask_buf),
                "profits": np.array(profit_buf),
                "action_names": action_names,
            })

        # Concatenate all trajectories for PPO update
        obs_all = np.concatenate([tr["obs"] for tr in trajectories], axis=0)
        acts_all = np.concatenate([tr["acts"] for tr in trajectories], axis=0)
        logp_all = np.concatenate([tr["logps"] for tr in trajectories], axis=0)
        rews_all = np.concatenate([tr["rews"] for tr in trajectories], axis=0)
        vals_all = np.concatenate([tr["vals"] for tr in trajectories], axis=0)
        masks_all = np.concatenate([tr["masks"] for tr in trajectories], axis=0)

        # compute returns and advantages with GAE
        returns, advs = compute_gae(rews_all, masks_all, vals_all, gamma=0.99, lam=0.95)

        # Convert to torch
        obs_t = torch.from_numpy(obs_all).float()
        acts_t = torch.from_numpy(acts_all).long()
        old_logps_t = torch.from_numpy(logp_all).float()
        returns_t = torch.from_numpy(returns).float()
        advs_t = torch.from_numpy(advs).float()

        # PPO update loop
        dataset_size = len(obs_all)
        inds = np.arange(dataset_size)
        for epoch in range(self.ppo_epochs):
            np.random.shuffle(inds)
            for start in range(0, dataset_size, self.batch_size):
                batch_idx = inds[start:start + self.batch_size]
                b_obs = obs_t[batch_idx].to(DEVICE)
                b_acts = acts_t[batch_idx].to(DEVICE)
                b_oldlogp = old_logps_t[batch_idx].to(DEVICE)
                b_returns = returns_t[batch_idx].to(DEVICE)
                b_advs = advs_t[batch_idx].to(DEVICE)

                logits = self.policy(b_obs)
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(b_acts)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logp - b_oldlogp)
                surr1 = ratio * b_advs
                surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * b_advs
                policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

                # value loss
                value_pred = self.value(b_obs).squeeze(-1)
                value_loss = (value_pred - b_returns).pow(2).mean()

                # update
                self.policy_optim.zero_grad()
                policy_loss.backward()
                self.policy_optim.step()

                self.value_optim.zero_grad()
                (self.value_coef * value_loss).backward()
                self.value_optim.step()

        # Save models
        self.save_models()

        # Finally, produce df_transaction for one *deterministic* run using the updated policy
        df_transaction = self._simulate_and_record(df, feats)

        return df_transaction

    def _simulate_and_record(self, df, feats) -> pd.DataFrame:
        """Run deterministic simulation (greedy from policy) and record Time, Price, Action, Profit for each tick."""
        self.load_models()  # load into policy & value with correct shapes
        T = len(df)
        env = TradingEnv(df["Price"].to_numpy(dtype=float))
        records = []
        for t in range(T):
            time = float(df.loc[t, "Time"])
            price = float(df.loc[t, "Price"])
            if t < WARMUP:
                action = 0
            else:
                obs = feats[t]
                # deterministic greedy
                x = torch.from_numpy(obs).float().unsqueeze(0).to(DEVICE)
                logits = self.policy(x)
                action = int(torch.argmax(logits, dim=-1).item())
                # enforce constraints
                if env.position.size > 0 and action in (1, 2):
                    action = 0
                if env.position.size == 0 and action == 3:
                    action = 0

            profit = 0.0
            if action == 1:  # BUY
                if env.can_open():
                    env.open_long(price)
            elif action == 2:
                if env.can_open():
                    env.open_short(price)
            elif action == 3:
                profit = env.repay(price)

            # forced repay at last tick
            if t == T - 1 and env.position.size > 0:
                profit = env.repay(price)
                action = 3

            records.append({
                "Time": time,
                "Price": price,
                "Action": ACTION_MAP[action],
                "Profit": profit,
            })
        df_out = pd.DataFrame.from_records(records)
        return df_out


# ---------------------------
# TradingSimulator class (inference-only for PC2)
# ---------------------------
class TradingSimulator:
    def __init__(self, policy_path="policy.pth", value_path="value.pth"):
        # model files must exist
        if not os.path.exists(policy_path) or not os.path.exists(value_path):
            raise FileNotFoundError("Model files not found. Please train and save policy/value models first.")
        # load shapes by probing with 1 dummy sample (we require obs_dim to be known by reading a small CSV or require user provide obs_dim)
        # For simplicity, we will load the policy by loading state_dict keys and inferring input dim from first Linear weight.
        self.policy_path = policy_path
        self.value_path = value_path
        # load state_dict to inspect input dim
        state = torch.load(policy_path, map_location=DEVICE)
        # assume first layer key like 'net.0.weight' exists
        first_w_key = next(k for k in state.keys() if "weight" in k)
        in_dim = state[first_w_key].shape[1]
        self.policy = PolicyNet(in_dim)
        self.value = ValueNet(in_dim)
        self.policy.load_state_dict(torch.load(policy_path, map_location=DEVICE))
        self.value.load_state_dict(torch.load(value_path, map_location=DEVICE))
        self.policy.to(DEVICE).eval()
        self.value.to(DEVICE).eval()

        # runtime buffers
        self.times = []
        self.prices = []
        self.volumes = []
        self.feats = None
        self.env = None
        self.last_action = 0  # default HOLD

    def _recompute_features(self):
        df = pd.DataFrame({"Time": self.times, "Price": self.prices, "Volume": self.volumes})
        self.feats = compute_features(df)
        if self.env is None:
            self.env = TradingEnv(np.array(self.prices, dtype=float))

    def add(self, time: float, price: float, volume: float) -> str:
        """
        Called every tick by user's app. Returns action string among "HOLD","BUY","SELL","REPAY".
        The TradingSimulator does NOT save ticks (the user's app does that); it only keeps runtime buffers needed for feature calculation.
        """
        self.times.append(time)
        self.prices.append(price)
        self.volumes.append(volume)
        # recompute features for all collected ticks (cheap for <= 20k ticks). Could be optimized to incremental.
        self._recompute_features()
        t = len(self.prices) - 1
        if t < WARMUP:
            self.last_action = 0
            return ACTION_MAP[0]

        obs = self.feats[t]
        x = torch.from_numpy(obs).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = self.policy(x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        # choose greedy action (could use some exploration flag if desired)
        action = int(np.argmax(probs))

        # enforce trading constraints
        if self.env.position.size > 0 and action in (1, 2):
            action = 0
        if self.env.position.size == 0 and action == 3:
            action = 0

        # execute corresponding env transitions to keep internal state consistent
        if action == 1:
            if self.env.can_open():
                self.env.open_long(price)
        elif action == 2:
            if self.env.can_open():
                self.env.open_short(price)
        elif action == 3:
            _ = self.env.repay(price)

        self.last_action = action
        return ACTION_MAP[action]


# ---------------------------
# Example: training loop usage (same structure as your reference)
# ---------------------------
if __name__ == "__main__":
    # Simple smoke test / usage example
    # Build a fake 1-day DataFrame if running standalone
    T = 1000
    times = np.arange(T).astype(float)
    # make synthetic price series
    prices = 1000 + np.cumsum(np.random.randn(T))
    volumes = np.cumsum(np.random.randint(1, 1000, size=T))
    df_day = pd.DataFrame({"Time": times, "Price": prices, "Volume": volumes})

    trainer = Trainer()
    df_trans = trainer.train(df_day)
    print("Sample trades summary:")
    print(df_trans[df_trans["Profit"] != 0.0].head())
    print("Total profit:", df_trans["Profit"].sum())

    # Example of inference usage:
    sim = TradingSimulator(policy_path="policy.pth", value_path="value.pth")
    # feed sequentially:
    for i in range(0, len(df_day)):
        a = sim.add(float(df_day.loc[i, "Time"]), float(df_day.loc[i, "Price"]), float(df_day.loc[i, "Volume"]))
        # in real use, your app records the tick to disk and also consumes `a` to execute orders
