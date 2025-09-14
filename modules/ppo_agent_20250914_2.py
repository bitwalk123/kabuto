import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from torch.distributions import Categorical

from modules.trading_env_20250914 import TradingEnv


# =========================================================
# Utility: Running mean/std for observation normalization
# =========================================================
class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


# =========================================================
# PPO Actor-Critic Network
# =========================================================
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, act_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        raise NotImplementedError

    def act(self, obs):
        logits = self.actor(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def evaluate(self, obs, act):
        logits = self.actor(obs)
        dist = Categorical(logits=logits)
        logp = dist.log_prob(act)
        entropy = dist.entropy()
        value = self.critic(obs).squeeze(-1)
        return logp, entropy, value


# =========================================================
# PPO Agent
# =========================================================
class PPOAgent:
    def __init__(self, obs_dim, act_dim, lr=3e-4, gamma=0.99, lam=0.95, clip_ratio=0.2,
                 ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, device="cpu"):
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self.device = device
        self.ac = ActorCritic(obs_dim, act_dim).to(device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)

    def compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0
        values = values + [0]
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.lam * (1 - dones[step]) * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return advantages, returns

    def update(self, obs, actions, log_probs_old, returns, advantages, epochs=10, batch_size=64):
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        log_probs_old = torch.tensor(log_probs_old, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = len(obs)
        for _ in range(epochs):
            idx = np.arange(dataset_size)
            np.random.shuffle(idx)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                mb_idx = idx[start:end]

                mb_obs = obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_log_probs_old = log_probs_old[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advantages = advantages[mb_idx]

                logp, entropy, value = self.ac.evaluate(mb_obs, mb_actions)

                ratio = torch.exp(logp - mb_log_probs_old)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = ((mb_returns - value) ** 2).mean()
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.optimizer.step()

        kl = (log_probs_old - logp).mean().item()
        clip_frac = ((ratio - 1.0).abs() > self.clip_ratio).float().mean().item()
        return kl, clip_frac


# =========================================================
# Training Loop
# =========================================================
def train(env, agent, epochs=100, steps_per_epoch=20000, device="cpu"):
    obs_rms = RunningMeanStd(shape=env.observation_space.shape)

    history = []
    for epoch in range(1, epochs + 1):
        obs, _ = env.reset()
        ep_reward = 0
        ep_pnl = 0

        obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []

        for step in range(steps_per_epoch):
            obs_rms.update(obs[None, :])
            obs_norm = (obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8)

            obs_tensor = torch.tensor(obs_norm, dtype=torch.float32, device=device)
            action, logp, _ = agent.ac.act(obs_tensor)
            value = agent.ac.critic(obs_tensor).item()

            next_obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated

            obs_buf.append(obs_norm)
            act_buf.append(action.item())
            logp_buf.append(logp.item())
            rew_buf.append(reward / 1000.0)  # 修正: 報酬を1000でスケーリング
            val_buf.append(value)
            done_buf.append(done)

            ep_reward += reward
            ep_pnl = info.get("pnl", ep_pnl)

            obs = next_obs
            if done:
                break

        advantages, returns = agent.compute_gae(rew_buf, val_buf, done_buf)
        kl, clip_frac = agent.update(obs_buf, act_buf, logp_buf, returns, advantages)

        history.append((epoch, step + 1, ep_reward, ep_pnl, kl, clip_frac))
        print(
            f"Epoch {epoch:03d} | Steps {step + 1} | Reward {ep_reward:.3f} | PnL {ep_pnl:.3f} | KL {kl:.6f} | ClipFrac {clip_frac:.4f}")

    df = pd.DataFrame(history, columns=["Epoch", "Steps", "Reward", "PnL", "KL", "ClipFrac"])
    df.to_csv("training_history.csv", index=False)
    print("Training finished. History saved to training_history.csv")


# =========================================================
# Entry Point
# =========================================================
if __name__ == "__main__":
    excel_path = "../excel/tick_20250819.xlsx"
    df = pd.read_excel(excel_path)
    env = TradingEnv(df)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = PPOAgent(obs_dim, act_dim, device=device)

    train(env, agent, epochs=100, steps_per_epoch=len(df) - 1, device=device)
