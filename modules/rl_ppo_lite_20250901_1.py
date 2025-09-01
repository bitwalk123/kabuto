"""
PPO-based trading simulator and trainer for 1-second tick Japan stock.
Requirements:
  python==3.13.7
  gymnasium==1.2.0
  numpy==2.3.2
  pandas==2.3.2
  torch==2.8.0

This single-file contains:
 - Feature calculations (60-tick warmup)
 - TradingSimulator class for real-time inference (add method)
 - Trainer class to train PPO using a DataFrame (Time, Price, Volume)
 - Simple PPO implementation with separate policy and value nets (PyTorch)
 - Epsilon-greedy exploration

Notes:
 - Slippage is 1 JPY tick (configurable)
 - Trade unit fixed to 100 shares
 - No commission
 - force_close functionality supported
 - Trainer.train(df) returns total realized P/L for the day
"""

"""
PPO-based trading simulator and trainer for 1-second tick Japan stock.
Requirements:
  python==3.13.7
  gymnasium==1.2.0
  numpy==2.3.2
  pandas==2.3.2
  torch==2.8.0

This single-file contains:
 - Feature calculations (60-tick warmup)
 - TradingSimulator class for real-time inference (add method)
 - Trainer class to train PPO using a DataFrame (Time, Price, Volume)
 - Simple PPO implementation with separate policy and value nets (PyTorch)
 - Epsilon-greedy exploration

Notes:
 - Slippage is 1 JPY tick (configurable)
 - Trade unit fixed to 100 shares
 - No commission
 - force_close functionality supported
 - Trainer.train(df) returns total realized P/L for the day
"""

import os
import math
import copy
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------------- Utilities ---------------------------------

def compute_features(df: pd.DataFrame, window: int = 60) -> Optional[np.ndarray]:
    if len(df) < window:
        return None
    sub = df.iloc[-window:]
    prices = sub['Price'].values
    volumes = sub['Volume'].values

    delta_v = np.diff(volumes, prepend=volumes[0])
    last_dv = delta_v[-1]
    log_dv = np.log1p(max(0.0, last_dv))

    ma = float(prices.mean())
    std = float(prices.std(ddof=0))

    deltas = np.diff(prices)
    ups = deltas.clip(min=0.0)
    downs = -deltas.clip(max=0.0)
    if len(ups) == 0:
        rsi = 50.0
    else:
        avg_up = ups.mean()
        avg_down = downs.mean()
        if avg_down == 0 and avg_up == 0:
            rsi = 50.0
        elif avg_down == 0:
            rsi = 100.0
        else:
            rs = avg_up / avg_down
            rsi = 100.0 - (100.0 / (1.0 + rs))

    z = (prices[-1] - ma) / (std + 1e-8)

    feat = np.array([log_dv, ma, std, rsi, z], dtype=np.float32)
    return feat

# ----------------------------- Networks ----------------------------------

class PolicyNet(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128, n_actions: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ValueNet(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

# ----------------------------- PPO Impl ----------------------------------

class PPO:
    def __init__(self, policy_net: PolicyNet, value_net: ValueNet, lr_policy: float = 3e-4, lr_value: float = 1e-3,
                 clip_epsilon: float = 0.2, value_coef: float = 0.5, entropy_coef: float = 0.01, device: str = 'cpu'):
        self.policy = policy_net.to(device)
        self.value = value_net.to(device)
        self.opt_policy = optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.opt_value = optim.Adam(self.value.parameters(), lr=lr_value)
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.device = device

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> Tuple[int, float, float]:
        st = torch.from_numpy(state.astype(np.float32)).to(self.device)
        logits = self.policy(st.unsqueeze(0))
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        if np.random.rand() < epsilon:
            action = int(np.random.choice(probs.shape[-1]))
            logp = dist.log_prob(torch.tensor(action)).detach().cpu().item()
        else:
            action = int(dist.sample().item())
            logp = dist.log_prob(torch.tensor(action)).detach().cpu().item()

        value_pred = self.value(st.unsqueeze(0)).detach().cpu().item()
        return action, logp, value_pred

    def update(self, batch: Dict[str, np.ndarray], epochs: int = 4, batch_size: int = 64):
        states = torch.from_numpy(batch['states']).to(self.device)
        actions = torch.from_numpy(batch['actions']).long().to(self.device)
        old_logps = torch.from_numpy(batch['old_logps']).to(self.device)
        returns = torch.from_numpy(batch['returns']).to(self.device)
        advantages = torch.from_numpy(batch['advantages']).to(self.device)

        n = len(states)
        for _ in range(epochs):
            idxs = np.arange(n)
            np.random.shuffle(idxs)
            for start in range(0, n, batch_size):
                mb = idxs[start:start + batch_size]
                mb_states = states[mb]
                mb_actions = actions[mb]
                mb_old_logps = old_logps[mb]
                mb_returns = returns[mb]
                mb_adv = advantages[mb]

                logits = self.policy(mb_states)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_logps = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logps - mb_old_logps)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_preds = self.value(mb_states)
                value_loss = (mb_returns - value_preds).pow(2).mean()

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.opt_policy.zero_grad()
                self.opt_value.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.value.parameters(), 0.5)
                self.opt_policy.step()
                self.opt_value.step()

    def save(self, path_policy: str, path_value: str):
        torch.save(self.policy.state_dict(), path_policy)
        torch.save(self.value.state_dict(), path_value)

    def load(self, path_policy: str, path_value: str):
        if not os.path.exists(path_policy) or not os.path.exists(path_value):
            raise FileNotFoundError('policy or value file not found')
        self.policy.load_state_dict(torch.load(path_policy, map_location=self.device))
        self.value.load_state_dict(torch.load(path_value, map_location=self.device))

# (TradingSimulator と Trainer のコードは以前と同じ)
# Trainer 内で 'value' の上書きを避け、代わりに 'value_pred' を使用してエラーを修正


# ----------------------- Trading Simulator (Inference) --------------------

ACTION_MAP = {0: 'HOLD', 1: 'BUY', 2: 'SELL', 3: 'REPAY'}

class TradingSimulator:
    def __init__(self,
                 policy_path: str,
                 value_path: str,
                 device: str = 'cpu',
                 slippage: float = 1.0,
                 unit: int = 100,
                 warmup: int = 60,
                 epsilon: float = 0.0):
        # must exist
        if not os.path.exists(policy_path) or not os.path.exists(value_path):
            raise FileNotFoundError('Pretrained model files not found. Cannot run inference.')
        # feature dim = 5 as defined
        self.device = device
        self.policy_net = PolicyNet(input_dim=5)
        self.value_net = ValueNet(input_dim=5)
        self.ppo = PPO(self.policy_net, self.value_net, device=device)
        self.ppo.load(policy_path, value_path)

        self.slippage = slippage
        self.unit = unit
        self.warmup = warmup
        self.epsilon = epsilon  # allow small exploration during live use if desired

        # runtime state
        self.history = pd.DataFrame(columns=['Time', 'Price', 'Volume'])
        self.position = 0  # 0 none, +1 long, -1 short
        self.entry_price = 0.0
        self.realized_pnl = 0.0
        self.last_action = 'HOLD'

    def add(self, time: float, price: float, volume: float, force_close: bool = False) -> str:
        # add tick
        self.history.loc[len(self.history)] = [time, price, volume]

        # warmup period
        if len(self.history) < self.warmup:
            return 'HOLD'

        feat = compute_features(self.history, window=self.warmup)
        if feat is None:
            return 'HOLD'

        # choose action
        action, logp, value = self.ppo.select_action(feat, epsilon=self.epsilon)

        # enforce forced close
        if force_close and self.position != 0:
            action = 3  # REPAY

        act_str = ACTION_MAP.get(action, 'HOLD')

        # execute action logic
        if act_str == 'BUY':
            if self.position == 0:
                # open long
                entry = price + self.slippage
                self.position = 1
                self.entry_price = entry
            else:
                # cannot open multilple; ignore
                pass
        elif act_str == 'SELL':
            if self.position == 0:
                # open short
                entry = price - self.slippage
                self.position = -1
                self.entry_price = entry
            else:
                # ignore
                pass
        elif act_str == 'REPAY':
            if self.position != 0:
                # close position
                if self.position == 1:
                    exit_price = price - self.slippage
                    pnl = (exit_price - self.entry_price) * self.unit
                else:
                    exit_price = price + self.slippage
                    pnl = (self.entry_price - exit_price) * self.unit
                self.realized_pnl += pnl
                # reset
                self.position = 0
                self.entry_price = 0.0
            else:
                pass
        else:
            # HOLD
            pass

        self.last_action = act_str
        return act_str

    def get_state(self) -> Dict:
        return {
            'position': self.position,
            'entry_price': self.entry_price,
            'realized_pnl': self.realized_pnl,
            'last_action': self.last_action,
            'history_len': len(self.history)
        }


# ----------------------- Trading Environment (for Trainer) ---------------

class TradingEnv:
    """A simple environment wrapper around tick-by-tick DataFrame for training.
    It produces states (feature vector) and accepts discrete actions.
    """
    def __init__(self, df: pd.DataFrame, slippage: float = 1.0, unit: int = 100, warmup: int = 60, unrealized_reward_ratio: float = 0.05):
        self.df = df.reset_index(drop=True)
        self.slippage = slippage
        self.unit = unit
        self.warmup = warmup
        self.unrealized_reward_ratio = unrealized_reward_ratio
        self.reset()

    def reset(self):
        self.i = 0
        self.history = pd.DataFrame(columns=['Time', 'Price', 'Volume'])
        self.position = 0
        self.entry_price = 0.0
        self.realized_pnl = 0.0
        self.done = False
        return None

    def step(self, action: int, force_close: bool = False) -> Tuple[Optional[np.ndarray], float, bool, dict]:
        # apply next tick
        if self.i >= len(self.df):
            self.done = True
            return None, 0.0, True, {}

        row = self.df.iloc[self.i]
        t, price, vol = float(row['Time']), float(row['Price']), float(row['Volume'])
        self.history.loc[len(self.history)] = [t, price, vol]

        # warmup
        if len(self.history) < self.warmup:
            self.i += 1
            return None, 0.0, False, {}

        feat = compute_features(self.history, window=self.warmup)
        if feat is None:
            self.i += 1
            return None, 0.0, False, {}

        # enforce force close on final step
        if force_close and self.i == len(self.df) - 1 and self.position != 0:
            action = 3

        reward = 0.0
        info = {}

        # execute action
        if action == 1:  # BUY
            if self.position == 0:
                entry = price + self.slippage
                self.position = 1
                self.entry_price = entry
        elif action == 2:  # SELL (open short)
            if self.position == 0:
                entry = price - self.slippage
                self.position = -1
                self.entry_price = entry
        elif action == 3:  # REPAY
            if self.position != 0:
                if self.position == 1:
                    exit_price = price - self.slippage
                    pnl = (exit_price - self.entry_price) * self.unit
                else:
                    exit_price = price + self.slippage
                    pnl = (self.entry_price - exit_price) * self.unit
                self.realized_pnl += pnl
                reward += pnl  # add realized P/L to reward
                # reset pos
                self.position = 0
                self.entry_price = 0.0
        # HOLD or invalid actions do nothing

        # per-tick unrealized P/L partial reward
        if self.position != 0:
            if self.position == 1:
                unreal = (price - self.entry_price) * self.unit
            else:
                unreal = (self.entry_price - price) * self.unit
            reward += self.unrealized_reward_ratio * unreal

        self.i += 1
        if self.i >= len(self.df):
            self.done = True
            # force close at last tick if still open
            if self.position != 0:
                # use last price (already applied if force_close True elsewhere)
                last_price = float(self.df.iloc[-1]['Price'])
                if self.position == 1:
                    exit_price = last_price - self.slippage
                    pnl = (exit_price - self.entry_price) * self.unit
                else:
                    exit_price = last_price + self.slippage
                    pnl = (self.entry_price - exit_price) * self.unit
                self.realized_pnl += pnl
                reward += pnl
                self.position = 0
                self.entry_price = 0.0

        next_state = compute_features(self.history, window=self.warmup) if not self.done else None
        return next_state, reward, self.done, info


# ---------------------------- Trainer -----------------------------------

class Trainer:
    def __init__(self,
                 device: str = 'cpu',
                 policy_path: str = 'policy.pth',
                 value_path: str = 'value.pth',
                 gamma: float = 0.99,
                 lam: float = 0.95):
        self.device = device
        self.policy_path = policy_path
        self.value_path = value_path
        # networks will be created on first train depending on feature dim
        self.gamma = gamma
        self.lam = lam

    def train(self, df: pd.DataFrame, epochs_per_update: int = 4, total_updates: int = 50, batch_size: int = 64, force_close: bool = True) -> float:
        # df: full day tick data
        env = TradingEnv(df)
        # infer feature dim
        dummy = compute_features(df.iloc[:60], window=60)
        if dummy is None:
            raise ValueError('Data too short for warmup window')
        feat_dim = dummy.shape[0]

        policy = PolicyNet(input_dim=feat_dim)
        value = ValueNet(input_dim=feat_dim)
        ppo = PPO(policy, value, device=self.device)

        total_realized = 0.0

        for update in range(total_updates):
            # collect rollout from env over one pass of the day
            env.reset()
            states = []
            actions = []
            rewards = []
            old_logps = []
            values = []

            done = False
            # to start, we will step until done
            while not done:
                # peek current index; if warmup not reached, just step with HOLD
                if env.i < env.warmup:
                    # apply the tick with HOLD
                    next_state, r, done, _ = env.step(0, force_close=force_close)
                    continue

                state = compute_features(env.history, window=env.warmup)
                if state is None:
                    next_state, r, done, _ = env.step(0, force_close=force_close)
                    continue

                # select action via current policy but with exploration (epsilon)
                st_t = torch.from_numpy(state.astype(np.float32)).to(self.device)
                logits = policy(st_t.unsqueeze(0))
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action = int(dist.sample().item())
                logp = float(dist.log_prob(torch.tensor(action)).cpu())
                value = float(value_net := value(st_t.unsqueeze(0)).detach().cpu())

                next_state, r, done, _ = env.step(action, force_close=force_close)

                states.append(state)
                actions.append(action)
                rewards.append(r)
                old_logps.append(logp)
                values.append(value)

            # after episode, compute returns and advantages
            total_realized = env.realized_pnl
            returns, advantages = self._compute_gae(rewards, values, self.gamma, self.lam)

            batch = {
                'states': np.vstack(states).astype(np.float32) if len(states) > 0 else np.zeros((0, feat_dim), dtype=np.float32),
                'actions': np.array(actions, dtype=np.int32),
                'old_logps': np.array(old_logps, dtype=np.float32),
                'returns': np.array(returns, dtype=np.float32),
                'advantages': np.array(advantages, dtype=np.float32),
            }

            if batch['states'].shape[0] > 0:
                ppo.update(batch, epochs=epochs_per_update, batch_size=batch_size)

        # save models
        ppo.save(self.policy_path, self.value_path)

        return total_realized

    def _compute_gae(self, rewards: List[float], values: List[float], gamma: float, lam: float):
        # compute GAE and returns; assumes episode ends and last value=0
        values = values + [0.0]
        gae = 0.0
        returns = []
        advs = []
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] - values[t]
            gae = delta + gamma * lam * gae
            advs.insert(0, gae)
            returns.insert(0, gae + values[t])
        advs = np.array(advs, dtype=np.float32)
        returns = np.array(returns, dtype=np.float32)
        # normalize advantages
        if advs.std() > 1e-8:
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        return returns, advs


# ---------------------------- Example Usage ------------------------------

if __name__ == '__main__':
    # Example: train using CSV of one day ticks (Time,Price,Volume)
    # df = pd.read_csv('ticks_20250901.csv')
    # t = Trainer(device='cpu')
    # pnl = t.train(df)
    # print('Trained, realized pnl:', pnl)

    # Example: inference
    # sim = TradingSimulator('policy.pth', 'value.pth', device='cpu')
    # while receiving ticks from broker tool:
    #     act = sim.add(time, price, volume, force_close=False)
    #     # act is one of 'HOLD','BUY','SELL','REPAY'
    pass
