import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np


# Policy Network: 離散的な行動空間を扱うため、出力は行動の確率分布
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        probs = self.softmax(self.fc3(x))
        return probs


# Value Network: 状態の価値を評価
class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        value = self.fc3(x)
        return value


class PPOAgent:
    def __init__(self, input_dim, output_dim, lr=3e-4, gamma=0.99, clip_epsilon=0.2, ppo_epochs=10):
        self.policy = PolicyNetwork(input_dim, output_dim)
        self.value = ValueNetwork(input_dim)
        self.optimizer_policy = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.optimizer_value = torch.optim.Adam(self.value.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs

    def get_action_and_value(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy(state_tensor)
        dist = Categorical(probs)
        action = dist.sample()
        value = self.value(state_tensor)
        return action.item(), dist.log_prob(action), value.item()

    def update(self, states, actions, old_log_probs, rewards, dones, next_states):
        # データのテンソル化
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        # 次の状態の価値を計算
        next_states = torch.FloatTensor(np.array(next_states))

        # 価値関数の推定
        values = self.value(states).squeeze()
        next_values = self.value(next_states).squeeze()

        # 割引報酬和 (GAE: Generalized Advantage Estimation) の計算
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * 0.95 * (1 - dones[t]) * last_gae

        # PPOの学習ループ
        for _ in range(self.ppo_epochs):
            # 新しいlog_probとvalueを計算
            new_probs = self.policy(states)
            new_dist = Categorical(new_probs)
            new_log_probs = new_dist.log_prob(actions)
            new_values = self.value(states).squeeze()

            # Policy Loss
            ratio = torch.exp(new_log_probs - old_log_probs.detach())
            surr1 = ratio * advantages.detach()
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages.detach()
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value Loss
            value_loss = (new_values - rewards).pow(2).mean()

            # Optimizerの更新
            self.optimizer_policy.zero_grad()
            policy_loss.backward()
            self.optimizer_policy.step()

            self.optimizer_value.zero_grad()
            value_loss.backward()
            self.optimizer_value.step()

    def save_model(self, path):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value.load_state_dict(checkpoint['value_state_dict'])