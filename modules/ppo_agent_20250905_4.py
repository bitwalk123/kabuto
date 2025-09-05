# ppo_agent.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# --------------------
# Actor-Critic Network
# --------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
        )
        self.policy = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        h = self.shared(x)
        logits = self.policy(h)
        value = self.value(h)
        return logits, value


# --------------------
# PPO Agent
# --------------------
class PPOAgent:
    def __init__(self, obs_dim, act_dim, lr=3e-4, gamma=0.99, clip_eps=0.2, ent_coef=0.01, vf_coef=0.5):
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = ActorCritic(obs_dim, act_dim).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

    def select_action(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.net(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action.item(), logprob.item(), value.item()

    def compute_returns(self, rewards, dones, last_value):
        """
        Monte-Carlo style discounted return
        """
        returns = []
        R = last_value
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d:
                R = 0
            R = r + self.gamma * R
            returns.insert(0, R)
        return returns

    def update(self, obs_buf, act_buf, old_logp_buf, ret_buf, adv_buf, epochs=4, batch_size=64):
        obs = torch.as_tensor(obs_buf, dtype=torch.float32, device=self.device)
        acts = torch.as_tensor(act_buf, dtype=torch.int64, device=self.device)
        old_logp = torch.as_tensor(old_logp_buf, dtype=torch.float32, device=self.device)
        rets = torch.as_tensor(ret_buf, dtype=torch.float32, device=self.device)
        advs = torch.as_tensor(adv_buf, dtype=torch.float32, device=self.device)

        dataset_size = len(obs)
        for _ in range(epochs):
            idxs = np.arange(dataset_size)
            np.random.shuffle(idxs)
            for start in range(0, dataset_size, batch_size):
                batch_idx = idxs[start:start + batch_size]
                b_obs, b_acts, b_old_logp, b_rets, b_advs = obs[batch_idx], acts[batch_idx], old_logp[batch_idx], rets[
                    batch_idx], advs[batch_idx]

                logits, values = self.net(b_obs)
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(b_acts)
                ratio = torch.exp(logp - b_old_logp)

                # policy loss
                clip_adv = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * b_advs
                loss_pi = -(torch.min(ratio * b_advs, clip_adv)).mean()

                # value loss
                loss_v = ((values.squeeze() - b_rets) ** 2).mean()

                # entropy bonus
                loss_ent = dist.entropy().mean()

                loss = loss_pi + self.vf_coef * loss_v - self.ent_coef * loss_ent

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
