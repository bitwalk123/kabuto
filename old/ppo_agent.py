"""
PPO trainer for the provided TradingEnv (gym.Env) using PyTorch.
Usage:
  - Put this file in the same folder as your TradingEnv class or adapt import path.
  - Ensure `tick_20250819.xlsx` is present in the working directory.
  - Install requirements: torch, gymnasium, numpy, pandas, ta-lib (or python-ta-lib wrapper), openpyxl

This trainer is tuned to learn reasonably fast on the provided environment:
 - actor-critic MLP with layer norm-ish initialization
 - GAE (lambda=0.95), gamma=0.99
 - PPO clipping (eps=0.2), value_ma loss coef=0.5, entropy coef=0.01
 - advantage normalization, value_ma target clipping
 - uses full-episode rollouts (one episode = full file) and performs multiple PPO epochs per episode

Notes:
 - This is a self-contained example for experimentation. You should adapt batch sizes / learning rates
   if you change the environment length or use vectorized envs.

"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


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
    def __init__(
            self,
            obs_dim: int,
            n_actions: int,
            hidden_sizes=(256, 256)
    ):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        self.shared = nn.Sequential(*layers)
        self.policy = nn.Sequential(
            nn.Linear(last, last // 2),
            nn.ReLU(),
            nn.Linear(last // 2, n_actions)
        )
        self.value = nn.Sequential(
            nn.Linear(last, last // 2),
            nn.ReLU(),
            nn.Linear(last // 2, 1)
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
def compute_gae(
        rewards,
        values,
        dones,
        last_value,
        gamma=0.99,
        lam=0.95
):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae_lam = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_non_terminal = 1.0 - dones[t]
            next_values = last_value
        else:
            next_non_terminal = 1.0 - dones[t + 1]
            next_values = values[t + 1]
        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
        advantages[t] = last_gae_lam = delta + gamma * lam * next_non_terminal * last_gae_lam
    returns = advantages + values
    return advantages, returns


def ppo_update(
        model: nn.Module,
        optimizer: optim.Optimizer,
        obs, actions,
        logprobs_old,
        returns,
        advantages,
        clip_coef=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
        epochs=8,
        minibatch_size=64
):
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

