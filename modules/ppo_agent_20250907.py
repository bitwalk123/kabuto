"""
PPO (PyTorch) — minimal, runnable example for the provided TradingEnv.
Assumptions:
- Your TradingEnv, ActionType enum are importable in the same namespace.
- Observation is a 1D numpy array, action space is Discrete(len(ActionType)).

Usage:
- Place this file in the same folder where TradingEnv is defined or adjust imports.
- Run: python ppo_agent_for_trading_env.py

This implementation is intentionally minimal but complete: a small MLP policy/value net,
rollout collection, GAE advantage, PPO clipping, minibatch updates.
Tune hyperparameters for your real experiments.
"""

import math
import random
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# Replace these imports if your env is defined elsewhere
# from your_module import TradingEnv, ActionType

# --- Minimal policy / value network ---
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_size: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_size, n_actions)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value


# --- Storage for rollouts ---
class RolloutBuffer:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def clear(self):
        self.__init__()


# --- PPO agent ---
class PPOAgent:
    def __init__(self,
                 obs_dim: int,
                 n_actions: int,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 lam: float = 0.95,
                 clip_eps: float = 0.2,
                 epochs: int = 4,
                 minibatch_size: int = 64,
                 device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = ActorCritic(obs_dim, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.minibatch_size = minibatch_size

    def select_action(self, obs: np.ndarray) -> Tuple[int, float, float]:
        obs_t = torch.from_numpy(obs.astype(np.float32)).to(self.device).unsqueeze(0)
        logits, value = self.net(obs_t)
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return int(action.item()), float(logprob.item()), float(value.item())

    def compute_gae(self, rewards, values, dones, last_value=0.0):
        # rewards, values, dones are lists
        advantages = []
        gae = 0.0
        values_extended = values + [last_value]
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values_extended[step + 1] * (1 - dones[step]) - values_extended[step]
            gae = delta + self.gamma * self.lam * (1 - dones[step]) * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns

    def update(self, buffer: RolloutBuffer, last_value: float = 0.0):
        advantages, returns = self.compute_gae(buffer.rewards, buffer.values, buffer.dones, last_value)
        obs_arr = np.array(buffer.obs, dtype=np.float32)
        actions_arr = np.array(buffer.actions, dtype=np.int64)
        old_logprobs_arr = np.array(buffer.logprobs, dtype=np.float32)
        advantages = np.array(advantages, dtype=np.float32)
        returns = np.array(returns, dtype=np.float32)

        # normalize advantages
        adv_mean, adv_std = advantages.mean(), advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        dataset_size = len(obs_arr)
        inds = np.arange(dataset_size)

        for _ in range(self.epochs):
            np.random.shuffle(inds)
            for start in range(0, dataset_size, self.minibatch_size):
                mb_inds = inds[start:start + self.minibatch_size]
                mb_obs = torch.from_numpy(obs_arr[mb_inds]).to(self.device)
                mb_actions = torch.from_numpy(actions_arr[mb_inds]).to(self.device)
                mb_old_logprobs = torch.from_numpy(old_logprobs_arr[mb_inds]).to(self.device)
                mb_advantages = torch.from_numpy(advantages[mb_inds]).to(self.device)
                mb_returns = torch.from_numpy(returns[mb_inds]).to(self.device)

                logits, values = self.net(mb_obs)
                probs = torch.softmax(logits, dim=-1)
                dist = Categorical(probs)
                mb_logprobs = dist.log_prob(mb_actions)

                ratio = torch.exp(mb_logprobs - mb_old_logprobs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = nn.functional.mse_loss(values, mb_returns)

                entropy = dist.entropy().mean()

                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.optimizer.step()
