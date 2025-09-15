"""
PPO trainer for the provided TradingEnv (gym.Env) using PyTorch.
Usage:
  - Put this file in the same folder as your TradingEnv class or adapt import path.
  - Ensure `tick_20250819.xlsx` is present in the working directory.
  - Install requirements: torch, gymnasium, numpy, pandas, ta-lib (or python-ta-lib wrapper), openpyxl

This trainer is tuned to learn reasonably fast on the provided environment:
 - actor-critic MLP with layer norm-ish initialization
 - GAE (lambda=0.95), gamma=0.99
 - PPO clipping (eps=0.2), value loss coef=0.5, entropy coef=0.01
 - advantage normalization, value target clipping
 - uses full-episode rollouts (one episode = full file) and performs multiple PPO epochs per episode

Notes:
 - This is a self-contained example for experimentation. You should adapt batch sizes / learning rates
   if you change the environment length or use vectorized envs.

"""

import os
import math
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from modules.trading_env_20250914 import TradingEnv


# If your TradingEnv is defined in another module, import it. Otherwise this file expects
# TradingEnv class to be available in the same Python path.
# from trading_env_module import TradingEnv

# ----------------------------- Utility: Running mean/std -----------------------------
class RunningMeanStd:
    def __init__(self, eps: float = 1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = eps

    def update(self, x: np.ndarray):
        x = np.array(x, dtype='float64')
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        if batch_count == 0:
            return
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / (tot_count)
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

# ----------------------------- ActorCritic Network -----------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes=(256, 256)):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        self.shared = nn.Sequential(*layers)
        self.policy = nn.Sequential(
            nn.Linear(last, last//2),
            nn.ReLU(),
            nn.Linear(last//2, n_actions)
        )
        self.value = nn.Sequential(
            nn.Linear(last, last//2),
            nn.ReLU(),
            nn.Linear(last//2, 1)
        )
        # init
        self.apply(self._weights_init)

    def _weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        shared = self.shared(x)
        logits = self.policy(shared)
        value = self.value(shared).squeeze(-1)
        return logits, value

# ----------------------------- PPO Trainer -----------------------------

def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae_lam = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_non_terminal = 1.0 - dones[t]
            next_values = last_value
        else:
            next_non_terminal = 1.0 - dones[t+1]
            next_values = values[t+1]
        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
        advantages[t] = last_gae_lam = delta + gamma * lam * next_non_terminal * last_gae_lam
    returns = advantages + values
    return advantages, returns


def ppo_update(model: nn.Module, optimizer: optim.Optimizer,
               obs, actions, logprobs_old, returns, advantages,
               clip_coef=0.2, vf_coef=0.5, ent_coef=0.01,
               max_grad_norm=0.5, epochs=8, minibatch_size=64):
    obs = torch.tensor(obs, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)
    logprobs_old = torch.tensor(logprobs_old, dtype=torch.float32)
    returns = torch.tensor(returns, dtype=torch.float32)
    advantages = torch.tensor(advantages, dtype=torch.float32)
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    dataset = TensorDataset(obs, actions, logprobs_old, returns, advantages)
    loader = DataLoader(dataset, batch_size=minibatch_size, shuffle=True)

    clip_frac = 0.0
    approx_kl = 0.0
    for _ in range(epochs):
        for batch in loader:
            b_obs, b_act, b_logp_old, b_ret, b_adv = batch
            logits, value = model(b_obs)
            dist = torch.distributions.Categorical(logits=logits)
            b_logp = dist.log_prob(b_act)
            ratio = torch.exp(b_logp - b_logp_old)
            surr1 = ratio * b_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * b_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            approx_kl += (b_logp_old - b_logp).mean().item()
            clip_frac += ((ratio > 1.0 + clip_coef) | (ratio < 1.0 - clip_coef)).float().mean().item()

            value_loss = (value - b_ret).pow(2).mean()
            entropy = dist.entropy().mean()

            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

    n_updates = epochs * (len(dataset) // minibatch_size + 1)
    return approx_kl / max(1, n_updates), clip_frac / max(1, n_updates)

# ----------------------------- Main training loop -----------------------------

def train_on_file(env_class, xlsx_path: str, n_epochs: int = 100, seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    df = pd.read_excel(xlsx_path)
    env = env_class(df)

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ActorCritic(obs_dim, n_actions).to(device)
    #optimizer = optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, eps=1e-5)

    # hyperparams (tuned to be stable for single-episode rollouts)
    gamma = 0.99
    lam = 0.95
    clip = 0.2
    ppo_epochs = 8
    minibatch_size = 128
    ent_coef = 0.01
    vf_coef = 0.5

    obs_rms = RunningMeanStd(shape=(obs_dim,))

    # storage for logs
    history = {
        'epoch': [], 'episode_reward': [], 'pnl_total': [], 'approx_kl': [], 'clipfrac': []
    }

    for epoch in range(1, n_epochs + 1):
        # run 1 full episode (one file -> one episode)
        obs_list, actions_list, rewards_list, dones_list, values_list, logp_list = [], [], [], [], [], []

        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        step = 0
        while not done:
            obs_norm = (obs - obs_rms.mean) / (np.sqrt(obs_rms.var) + 1e-8)
            obs_tensor = torch.tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, value = model(obs_tensor)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample().cpu().numpy()[0]
                logp = dist.log_prob(torch.tensor(action)).cpu().numpy()
                value = value.cpu().numpy()[0]

            next_obs, reward, done, truncated, info = env.step(int(action))

            obs_list.append(obs)
            actions_list.append(action)
            rewards_list.append(reward)
            dones_list.append(done)
            values_list.append(value)
            logp_list.append(logp)

            total_reward += reward
            obs = next_obs
            step += 1

        # update running obs stats
        obs_rms.update(np.array(obs_list))

        # compute last value for bootstrap
        with torch.no_grad():
            last_obs_norm = (obs - obs_rms.mean) / (np.sqrt(obs_rms.var) + 1e-8)
            last_obs_t = torch.tensor(last_obs_norm, dtype=torch.float32, device=device).unsqueeze(0)
            _, last_value = model(last_obs_t)
            last_value = last_value.cpu().numpy()[0]

        # compute GAE advantages and returns
        values_arr = np.asarray(values_list, dtype=np.float32)
        advantages, returns = compute_gae(np.asarray(rewards_list, dtype=np.float32), values_arr, np.asarray(dones_list, dtype=np.float32), last_value, gamma=gamma, lam=lam)

        # prepare logprobs_old as float array
        logp_arr = np.asarray(logp_list, dtype=np.float32)

        # normalize observations at training time
        obs_arr = np.asarray([(o - obs_rms.mean) / (np.sqrt(obs_rms.var) + 1e-8) for o in obs_list], dtype=np.float32)

        # PPO update
        approx_kl, clipfrac = ppo_update(model, optimizer,
                                         obs_arr, actions_list, logp_arr, returns, advantages,
                                         clip_coef=clip, vf_coef=vf_coef, ent_coef=ent_coef,
                                         max_grad_norm=0.5, epochs=ppo_epochs, minibatch_size=minibatch_size)

        history['epoch'].append(epoch)
        history['episode_reward'].append(total_reward)
        history['pnl_total'].append(env.transman.pnl_total)
        history['approx_kl'].append(approx_kl)
        history['clipfrac'].append(clipfrac)

        print(f"Epoch {epoch:03d} | Steps {step:05d} | Reward {total_reward:.3f} | PnL {env.transman.pnl_total:.3f} | KL {approx_kl:.6f} | ClipFrac {clipfrac:.4f}")

        # save model every 10 epochs
        if epoch % 10 == 0:
            fname = f"ppo_trading_epoch{epoch}.pt"
            torch.save(model.state_dict(), fname)

    # final save
    torch.save(model.state_dict(), 'ppo_trading_final.pt')

    # write history to CSV
    hist_df = pd.DataFrame(history)
    hist_df.to_csv('training_history.csv', index=False)
    print('\nTraining finished. History saved to training_history.csv')


if __name__ == '__main__':
    # If your TradingEnv is defined in this runtime, import it directly. Otherwise adjust the path.
    # For example, if TradingEnv is defined in trading_env.py: from trading_env import TradingEnv
    """
    try:
        # try to import TradingEnv from the global scope
        from __main__ import TradingEnv  # if TradingEnv defined in same execution context
    except Exception:
        # fallback: try to import from a module named trading_env
        try:
            from trading_env import TradingEnv
        except Exception:
            raise ImportError('TradingEnv not found. Put TradingEnv class in the same file or adjust import path.')
    """

    # Path to your tick data file
    xlsx = '../excel/tick_20250828.xlsx'
    if not os.path.exists(xlsx):
        raise FileNotFoundError(f"{xlsx} not found in working directory")

    train_on_file(TradingEnv, xlsx, n_epochs=100, seed=12345)
