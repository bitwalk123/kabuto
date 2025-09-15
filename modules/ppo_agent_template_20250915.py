import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

from modules.ppo_agent_20250914_1 import ActorCritic


class PPOAgent:
    def __init__(
            self,
            obs_dim,
            n_actions,
            lr=3e-4,
            gamma=0.99,
            lam=0.95,
            clip_eps=0.2,
            n_epochs=10,
            batch_size=64,
            hidden_sizes=(256, 256)
    ):
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # ネットワーク
        self.model = ActorCritic(obs_dim, n_actions, hidden_sizes)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # バッファ
        self.reset_buffer()

    def reset_buffer(self):
        self.obs_buf, self.act_buf, self.logp_buf = [], [], []
        self.rew_buf, self.done_buf, self.val_buf = [], [], []

    def select_action(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits, value = self.model(obs_t)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        return action.item(), logp.item(), value.item()

    def store_transition(self, obs, act, logp, rew, done, val):
        self.obs_buf.append(obs)
        self.act_buf.append(act)
        self.logp_buf.append(logp)
        self.rew_buf.append(rew)
        self.done_buf.append(done)
        self.val_buf.append(val)

    def finish_trajectory(self, last_val=0):
        """GAE-Lambda 計算 & バッファ整形"""
        rews = np.array(self.rew_buf + [last_val])
        vals = np.array(self.val_buf + [last_val])

        adv = np.zeros_like(self.rew_buf, dtype=np.float32)
        gae = 0
        for t in reversed(range(len(self.rew_buf))):
            delta = rews[t] + self.gamma * vals[t + 1] * (1 - self.done_buf[t]) - vals[t]
            gae = delta + self.gamma * self.lam * (1 - self.done_buf[t]) * gae
            adv[t] = gae

        ret = adv + np.array(self.val_buf, dtype=np.float32)

        self.obs_buf = torch.as_tensor(self.obs_buf, dtype=torch.float32)
        self.act_buf = torch.as_tensor(self.act_buf, dtype=torch.int64)
        self.logp_buf = torch.as_tensor(self.logp_buf, dtype=torch.float32)
        self.adv_buf = torch.as_tensor((adv - adv.mean()) / (adv.std() + 1e-8), dtype=torch.float32)
        self.ret_buf = torch.as_tensor(ret, dtype=torch.float32)

    def update(self):
        dataset = torch.utils.data.TensorDataset(
            self.obs_buf,
            self.act_buf,
            self.logp_buf,
            self.adv_buf,
            self.ret_buf
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for _ in range(self.n_epochs):
            for obs, act, logp_old, adv, ret in loader:
                logits, value = self.model(obs)
                dist = Categorical(logits=logits)
                logp = dist.log_prob(act)

                ratio = torch.exp(logp - logp_old)

                # PPO Clip 損失
                policy_loss = -torch.min(
                    ratio * adv,
                    torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
                ).mean()

                # Value 損失
                value_loss = ((ret - value) ** 2).mean()

                # 損失を合成
                loss = policy_loss + 0.5 * value_loss - 0.01 * dist.entropy().mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

        self.reset_buffer()
