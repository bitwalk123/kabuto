"""
Simplified PPO agent for TradingEnv (PyTorch implementation)

- Designed for CPU training on tick-level TradingEnv
- Compatible with gymnasium environment interface
- Focused on clarity and minimal implementation rather than maximum efficiency

Dependencies:
    torch==2.8.0
    numpy==2.3.2
    pandas==2.3.2
    gymnasium==1.2.0

Usage:
    from TradingEnv import TradingEnv, ActionType
    import pandas as pd
    import numpy as np

    # Prepare dummy data (replace with real preprocessed tick data)
    df = pd.DataFrame({
        'Time': pd.date_range('2025-01-01', periods=500, freq='s'),
        'Price': 1000 + np.cumsum(np.random.randn(500)),
        'Volume': np.cumsum(np.random.poisson(5, 500))
    })
    env = TradingEnv(df)

    agent = PPOAgent(env)
    agent.train(num_episodes=10)
    agent.save("model_latest.pt")
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, act_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        features = self.net(x)
        logits = self.policy_head(features)
        value = self.value_head(features)
        return logits, value


class PPOAgent:
    def __init__(
            self,
            env,
            gamma: float = 0.99,
            lam: float = 0.95,
            clip_ratio: float = 0.2,
            lr: float = 3e-4,
            update_epochs: int = 10,
            batch_size: int = 64,
            device: str = "cpu",
    ):
        self.env = env
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n

        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.device = device

        self.policy = PolicyNetwork(obs_dim, act_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def select_action(self, obs):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        logits, value = self.policy(obs_t)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

    def compute_gae(self, rewards, values, dones, next_value):
        advs = []
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * (1 - dones[t]) * next_value - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advs.insert(0, gae)
            next_value = values[t]
        returns = [a + v for a, v in zip(advs, values)]
        return np.array(advs, dtype=np.float32), np.array(returns, dtype=np.float32)

    def update(self, obs_buf, act_buf, logp_buf, adv_buf, ret_buf):
        obs_t = torch.tensor(obs_buf, dtype=torch.float32).to(self.device)
        act_t = torch.tensor(act_buf, dtype=torch.int64).to(self.device)
        old_logp_t = torch.tensor(logp_buf, dtype=torch.float32).to(self.device)
        adv_t = torch.tensor(adv_buf, dtype=torch.float32).to(self.device)
        ret_t = torch.tensor(ret_buf, dtype=torch.float32).to(self.device)

        dataset_size = len(obs_buf)
        for _ in range(self.update_epochs):
            idxs = np.arange(dataset_size)
            np.random.shuffle(idxs)
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_idx = idxs[start:end]

                logits, values = self.policy(obs_t[batch_idx])
                dist = Categorical(logits=logits)
                logp = dist.log_prob(act_t[batch_idx])
                ratio = torch.exp(logp - old_logp_t[batch_idx])

                # Policy loss (clipped surrogate)
                surr1 = ratio * adv_t[batch_idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv_t[batch_idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (MSE)
                value_loss = ((values.squeeze() - ret_t[batch_idx]) ** 2).mean()

                # Entropy bonus
                entropy = dist.entropy().mean()
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def train(self, num_episodes: int = 100):
        for ep in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            ep_reward = 0

            obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []

            while not done:
                action, logp, value = self.select_action(obs)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                obs_buf.append(obs)
                act_buf.append(action)
                logp_buf.append(logp)
                rew_buf.append(reward)
                val_buf.append(value)
                done_buf.append(float(done))

                obs = next_obs
                ep_reward += reward

            # bootstrap value
            with torch.no_grad():
                next_obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                _, next_value = self.policy(next_obs_t)
                next_value = next_value.item()

            adv_buf, ret_buf = self.compute_gae(rew_buf, val_buf, done_buf, next_value)

            self.update(obs_buf, act_buf, logp_buf, adv_buf, ret_buf)

            print(f"Episode {ep + 1}/{num_episodes}, Reward: {ep_reward:.2f}")

    def save(self, path: str):
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
        self.policy.eval()
