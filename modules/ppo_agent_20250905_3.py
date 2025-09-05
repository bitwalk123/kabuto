import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from modules.trading_env_20250925_3 import TradingEnv


# -------------------------
# Actor-Critic ネットワーク
# -------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=128):
        super().__init__()
        # Actor
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Softmax(dim=-1),
        )
        # Critic
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs):
        policy = self.actor(obs)
        value = self.critic(obs)
        return policy, value

# -------------------------
# PPO エージェント
# -------------------------
class PPOAgent:
    def __init__(self, obs_dim, act_dim, gamma=0.99, clip_eps=0.2, lr=3e-4, device="cpu"):
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.device = device

        self.net = ActorCritic(obs_dim, act_dim).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

    def select_action(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device).unsqueeze(0)
        probs, value = self.net(obs)
        dist = Categorical(probs)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action.item(), logprob.item(), value.item()

    def compute_returns(self, rewards, dones, last_value):
        returns = []
        R = last_value
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d:
                R = 0  # episode 終了時にリセット
            R = r + self.gamma * R
            returns.insert(0, R)
        return returns

    def update(self, memory, batch_size=32, epochs=4):
        obs = torch.tensor(np.array(memory["obs"]), dtype=torch.float32).to(self.device)
        actions = torch.tensor(memory["actions"]).to(self.device)
        old_logprobs = torch.tensor(memory["logprobs"], dtype=torch.float32).to(self.device)
        returns = torch.tensor(memory["returns"], dtype=torch.float32).to(self.device)

        for _ in range(epochs):
            idx = np.random.permutation(len(obs))
            for i in range(0, len(obs), batch_size):
                batch_idx = idx[i:i+batch_size]

                batch_obs = obs[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_logprobs = old_logprobs[batch_idx]
                batch_returns = returns[batch_idx]

                probs, values = self.net(batch_obs)
                dist = Categorical(probs)
                entropy = dist.entropy().mean()
                new_logprobs = dist.log_prob(batch_actions)
                values = values.squeeze()

                advantages = batch_returns - values.detach()

                ratio = torch.exp(new_logprobs - batch_old_logprobs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(values, batch_returns)
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

# -------------------------
# スモークテスト
# -------------------------
if __name__ == "__main__":
    # Excel からデータ読込
    df = pd.read_excel("../excel/tick_20250828.xlsx")
    env = TradingEnv(df)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    agent = PPOAgent(obs_dim, act_dim, device="cpu")

    n_episodes = 100  # スモークテストなので少なめ
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        memory = {"obs": [], "actions": [], "logprobs": [], "rewards": [], "dones": [], "returns": []}
        total_reward = 0

        while not done:
            action, logprob, value = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            memory["obs"].append(obs)
            memory["actions"].append(action)
            memory["logprobs"].append(logprob)
            memory["rewards"].append(reward)
            memory["dones"].append(done)

            obs = next_obs
            total_reward += reward

        # GAE ではなく単純 discounted return で OK
        last_value = 0
        returns = agent.compute_returns(memory["rewards"], memory["dones"], last_value)
        memory["returns"] = returns

        # PPO update
        agent.update(memory)

        print(f"Episode {ep+1}/{n_episodes}, TotalReward={total_reward:.2f}, RealizedPnL={env.realized_pnl:.2f}")
