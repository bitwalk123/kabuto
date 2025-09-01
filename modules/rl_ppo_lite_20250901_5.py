# rl_trading.py
import os
import math
import copy
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque, namedtuple

# ----------------------------
# 環境/取引ロジックの定義
# ----------------------------
ActionMap = {0: "HOLD", 1: "BUY", 2: "SELL", 3: "REPAY"}


class Position:
    def __init__(self):
        self.side = None  # 'long' or 'short' or None
        self.entry_price = None
        self.entry_time = None

    def is_open(self):
        return self.side is not None

    def open_long(self, price, time):
        self.side = 'long'
        self.entry_price = price
        self.entry_time = time

    def open_short(self, price, time):
        self.side = 'short'
        self.entry_price = price
        self.entry_time = time

    def close(self):
        self.side = None
        self.entry_price = None
        self.entry_time = None


# ----------------------------
# 指標計算ヘルパー
# ----------------------------
def compute_rsi(prices: np.ndarray, n: int = 60) -> float:
    # prices: 1D np array, len >= n
    if len(prices) < n + 1:
        return 50.0
    delta = np.diff(prices[-(n + 1):])
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_gain = gains.mean()
    avg_loss = losses.mean()
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi)


# ----------------------------
# ネットワーク
# ----------------------------
class PolicyNet(nn.Module):
    def __init__(self, obs_dim, hidden=128, n_actions=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, x):
        return self.net(x)  # ログits（softmaxは外で）


class ValueNet(nn.Module):
    def __init__(self, obs_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ----------------------------
# PPO Agent
# ----------------------------
Transition = namedtuple('Transition', ['obs', 'action', 'logp', 'reward', 'done', 'value'])


class PPOAgent:
    def __init__(self,
                 obs_dim: int,
                 n_actions: int = 4,
                 device: str = 'cpu',
                 lr_policy: float = 3e-4,
                 lr_value: float = 1e-3,
                 clip_epsilon: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 1e-3,
                 ppo_epochs: int = 4,
                 minibatch_size: int = 64):
        self.device = torch.device(device)
        self.policy = PolicyNet(obs_dim, n_actions=n_actions).to(self.device)
        self.value = ValueNet(obs_dim).to(self.device)
        self.opt_policy = optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.opt_value = optim.Adam(self.value.parameters(), lr=lr_value)
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size

    def act(self, obs: np.ndarray, legal_mask: Optional[np.ndarray] = None, epsilon: float = 0.0) -> Tuple[
        int, float, float]:
        """obs: 1D array. legal_mask: boolean array of shape (n_actions,) True if allowed.
           epsilon: epsilon-greedy probability to pick a random legal action.
           returns action, logp, value
        """
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.policy(obs_t).detach().cpu().numpy().flatten()
        if legal_mask is not None:
            # illegal actions get large negative logit
            logits = np.where(legal_mask, logits, -1e8)
        # epsilon-greedy
        legal_actions = np.where(legal_mask)[0] if legal_mask is not None else np.arange(len(logits))
        if (epsilon > 0.0) and (random.random() < epsilon):
            action = int(np.random.choice(legal_actions))
            # compute logp via softmax for returned action (approx)
            probs = np.exp(logits - logits.max())
            probs = probs / probs.sum()
            logp = float(np.log(probs[action] + 1e-12))
        else:
            # sample from categorical
            probs = np.exp(logits - logits.max())
            probs = probs / probs.sum()
            action = int(np.random.choice(len(probs), p=probs))
            logp = float(np.log(probs[action] + 1e-12))
        with torch.no_grad():
            value = float(self.value(obs_t).cpu().numpy().flatten()[0])
        return action, logp, value

    def save(self, policy_path: str, value_path: Optional[str] = None):
        torch.save(self.policy.state_dict(), policy_path)
        if value_path:
            torch.save(self.value.state_dict(), value_path)

    def load(self, policy_path: str, value_path: Optional[str] = None):
        if not os.path.exists(policy_path):
            raise FileNotFoundError(policy_path)
        self.policy.load_state_dict(torch.load(policy_path, map_location=self.device))
        if value_path:
            if not os.path.exists(value_path):
                raise FileNotFoundError(value_path)
            self.value.load_state_dict(torch.load(value_path, map_location=self.device))

    def ppo_update(self, transitions: List[Transition], gamma: float = 0.999, lam: float = 0.95):
        """Simplified on-policy PPO update. transitions is a list of Transition collected sequentially."""
        # Convert to arrays
        obs = torch.tensor(np.vstack([t.obs for t in transitions]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([t.action for t in transitions], dtype=torch.long, device=self.device)
        old_logp = torch.tensor([t.logp for t in transitions], dtype=torch.float32, device=self.device)
        rewards = np.array([t.reward for t in transitions], dtype=np.float32)
        dones = np.array([t.done for t in transitions], dtype=np.float32)
        values = np.array([t.value for t in transitions], dtype=np.float32)

        # Compute returns and advantages (GAE)
        T = len(rewards)
        returns = np.zeros(T, dtype=np.float32)
        advs = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        last_value = 0.0
        for t in reversed(range(T)):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * last_value * nonterminal - values[t]
            last_gae = delta + gamma * lam * nonterminal * last_gae
            advs[t] = last_gae
            returns[t] = advs[t] + values[t]
            last_value = values[t]

        advs = torch.tensor(advs, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # PPO epochs
        dataset_size = len(transitions)
        for _ in range(self.ppo_epochs):
            idxs = np.arange(dataset_size)
            np.random.shuffle(idxs)
            for start in range(0, dataset_size, self.minibatch_size):
                mb_idx = idxs[start:start + self.minibatch_size]
                mb_obs = obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_logp = old_logp[mb_idx]
                mb_adv = advs[mb_idx]
                mb_returns = returns[mb_idx]

                logits = self.policy(mb_obs)
                dist = F.softmax(logits, dim=-1)
                mb_logp = torch.log(dist[range(len(mb_actions)), mb_actions] + 1e-12)
                ratio = torch.exp(mb_logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_adv
                policy_loss = -torch.mean(torch.min(surr1, surr2))

                entropy = -torch.mean(torch.sum(dist * torch.log(dist + 1e-12), dim=-1))

                self.opt_policy.zero_grad()
                (policy_loss - self.entropy_coef * entropy).backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.opt_policy.step()

                # value loss
                value_pred = self.value(mb_obs)
                value_loss = F.mse_loss(value_pred, mb_returns)
                self.opt_value.zero_grad()
                (self.value_coef * value_loss).backward()
                nn.utils.clip_grad_norm_(self.value.parameters(), 0.5)
                self.opt_value.step()


# ----------------------------
# TradingSimulator（推論用）
# ----------------------------
class TradingSimulator:
    def __init__(self,
                 model_path: str,
                 device: str = 'cpu',
                 unit_size: int = 100,
                 slippage: float = 1.0,
                 warmup: int = 60,
                 epsilon: float = 0.0):
        """
        model_path: path to policy.pth (required)
        epsilon: epsilon-greedy probability for inference (small)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Policy model not found: {model_path}")
        # obs_dim: we'll define obs as [price_norm, ma, std, rsi, zscore, log1p_dvol, position_flag]
        self.obs_dim = 7
        self.agent = PPOAgent(obs_dim=self.obs_dim, device=device)
        self.agent.load(policy_path=model_path)
        self.unit = unit_size
        self.slippage = slippage
        self.warmup = warmup
        self.epsilon = epsilon
        self.reset()

    def reset(self):
        self.prices = deque(maxlen=self.warmup)
        self.vols = deque(maxlen=self.warmup + 1)  # 累計ボリュームの差分計算用
        self.position = Position()
        self.realized_pnl = 0.0
        self.last_time = None

    def _compute_features(self, price: float, volume: float):
        # maintain price and volume windows
        self.prices.append(price)
        self.vols.append(volume)
        if len(self.prices) < self.warmup:
            return None  # warmup
        prices_arr = np.array(self.prices)
        vols_arr = np.array(self.vols)
        # ΔVolume: difference of cumulative vol. If negative (shouldn't), clip to 0
        if len(vols_arr) >= 2:
            dvol = vols_arr[-1] - vols_arr[-2]
        else:
            dvol = 0.0
        log1p_dvol = float(np.log1p(max(dvol, 0.0)))
        ma = float(prices_arr.mean())
        std = float(prices_arr.std(ddof=0) + 1e-8)
        rsi = compute_rsi(prices_arr, n=self.warmup)
        zscore = float((price - ma) / std)
        position_flag = 0.0
        if self.position.is_open():
            position_flag = 1.0 if self.position.side == 'long' else -1.0
        # normalize price by ma to keep scale small
        price_norm = float(price / (ma + 1e-8) - 1.0)
        obs = np.array([price_norm, ma, std, rsi, zscore, log1p_dvol, position_flag], dtype=np.float32)
        return obs

    def _legal_actions_mask(self):
        # returns boolean mask length 4 for actions [HOLD, BUY, SELL, REPAY]
        mask = np.array([True, True, True, True], dtype=bool)
        if self.position.is_open():
            # if long or short, cannot enter new position (no pyramiding)
            mask[1] = False  # BUY
            mask[2] = False  # SELL
            # HOLD and REPAY allowed
        else:
            # no position: REPAY not allowed
            mask[3] = False
        return mask

    def add(self, time: float, price: float, volume: float) -> str:
        """
        Called per tick. Returns action string among "HOLD","BUY","SELL","REPAY".
        Must load model at initialization (done in __init__), otherwise FileNotFoundError.
        """
        self.last_time = time
        obs = self._compute_features(price, volume)
        if obs is None:
            return "HOLD"  # warmup
        legal_mask = self._legal_actions_mask()
        action, logp, value = self.agent.act(obs, legal_mask=legal_mask, epsilon=self.epsilon)
        act_str = ActionMap[action]

        # Enforce allowed actions again (in case policy chose illegal due to numeric)
        if not legal_mask[action]:
            # fallback to HOLD if chosen illegal
            action = 0
            act_str = "HOLD"

        # Execute trade logic
        reward = 0.0
        # For new entry, entry price includes slippage
        if action == 1 and not self.position.is_open():  # BUY
            entry = price + self.slippage
            self.position.open_long(entry, time)
            # no immediate realized PnL
        elif action == 2 and not self.position.is_open():  # SELL (enter short)
            entry = price - self.slippage
            self.position.open_short(entry, time)
        elif action == 3 and self.position.is_open():  # REPAY
            if self.position.side == 'long':
                exit_price = price - self.slippage
                profit = (exit_price - self.position.entry_price) * self.unit
            else:
                exit_price = price + self.slippage
                profit = (self.position.entry_price - exit_price) * self.unit
            self.realized_pnl += profit
            reward += profit  # reward on realized PnL
            self.position.close()
        # HOLD gives no realized reward, but include unrealized small reward per tick
        # Provide 5% of unrealized PnL per tick as reward if holding
        if self.position.is_open():
            if self.position.side == 'long':
                unreal = (price - self.position.entry_price) * self.unit
            else:
                unreal = (self.position.entry_price - price) * self.unit
            reward += 0.05 * unreal  # per-tick small shaping reward

        # Return action string (user's requirement)
        return act_str


# ----------------------------
# Trainer（学習用）
# ----------------------------
class Trainer:
    def __init__(self,
                 model_path: str = 'policy.pth',
                 value_path: str = 'value.pth',
                 device: str = 'cpu',
                 unit_size: int = 100,
                 slippage: float = 1.0,
                 warmup: int = 60,
                 gamma: float = 0.999,
                 epsilon_start: float = 0.2,
                 epsilon_end: float = 0.02,
                 epsilon_decay_steps: int = 10000):
        self.model_path = model_path
        self.value_path = value_path
        self.device = device
        self.unit = unit_size
        self.slippage = slippage
        self.warmup = warmup
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

        self.obs_dim = 7
        self.agent = PPOAgent(obs_dim=self.obs_dim, device=device)
        self.reset()

    def reset(self):
        self.prices = deque(maxlen=self.warmup)
        self.vols = deque(maxlen=self.warmup + 1)
        self.position = Position()
        self.realized_pnl = 0.0
        self.transactions = []  # records for df_transaction as tuples (Time, Price, Action, Profit)
        self.transitions = []  # for PPO

    def _compute_features(self, price: float, volume: float):
        self.prices.append(price)
        self.vols.append(volume)
        if len(self.prices) < self.warmup:
            return None
        prices_arr = np.array(self.prices)
        vols_arr = np.array(self.vols)
        if len(vols_arr) >= 2:
            dvol = vols_arr[-1] - vols_arr[-2]
        else:
            dvol = 0.0
        log1p_dvol = float(np.log1p(max(dvol, 0.0)))
        ma = float(prices_arr.mean())
        std = float(prices_arr.std(ddof=0) + 1e-8)
        rsi = compute_rsi(prices_arr, n=self.warmup)
        zscore = float((price - ma) / std)
        position_flag = 0.0
        if self.position.is_open():
            position_flag = 1.0 if self.position.side == 'long' else -1.0
        price_norm = float(price / (ma + 1e-8) - 1.0)
        obs = np.array([price_norm, ma, std, rsi, zscore, log1p_dvol, position_flag], dtype=np.float32)
        return obs

    def _legal_actions_mask(self):
        mask = np.array([True, True, True, True], dtype=bool)
        if self.position.is_open():
            mask[1] = False
            mask[2] = False
        else:
            mask[3] = False
        return mask

    def train(self, df: pd.DataFrame, epochs: int = 1, ppo_update_after_ticks: int = 200):
        """
        df: DataFrame with columns Time, Price, Volume (cumulative)
        Returns df_transaction with Time, Price, Action, Profit (profit filled when REPAY else 0)
        """
        assert {'Time', 'Price', 'Volume'}.issubset(df.columns), "df must contain Time, Price, Volume"
        self.reset()
        total_steps = len(df)
        step_count = 0
        epsilon = self.epsilon_start
        df_transactions = []
        # Iterate through ticks
        for idx, row in df.iterrows():
            t = float(row['Time'])
            price = float(row['Price'])
            vol = float(row['Volume'])
            obs = self._compute_features(price, vol)
            step_count += 1
            # Warmup: force HOLD
            if obs is None:
                action = 0
                reward = 0.0
                value = 0.0
                logp = 0.0
                done = False
                # record transaction
                df_transactions.append({'Time': t, 'Price': price, 'Action': ActionMap[action], 'Profit': 0.0})
                # no transition recorded during warmup
                continue

            # compute legal actions mask
            legal_mask = self._legal_actions_mask()
            # epsilon decay
            frac = min(1.0, step_count / max(1, self.epsilon_decay_steps))
            epsilon = self.epsilon_start * (1 - frac) + self.epsilon_end * frac

            # act using current policy with epsilon-greedy
            action, logp, value = self.agent.act(obs, legal_mask=legal_mask, epsilon=epsilon)
            # guard: if illegal, force HOLD
            if not legal_mask[action]:
                action = 0

            reward = 0.0
            profit_record = 0.0
            # Execute trades
            if action == 1 and not self.position.is_open():  # BUY
                entry = price + self.slippage
                self.position.open_long(entry, t)
                profit_record = 0.0
            elif action == 2 and not self.position.is_open():  # SELL (open short)
                entry = price - self.slippage
                self.position.open_short(entry, t)
                profit_record = 0.0
            elif action == 3 and self.position.is_open():  # REPAY
                if self.position.side == 'long':
                    exit_price = price - self.slippage
                    profit = (exit_price - self.position.entry_price) * self.unit
                else:
                    exit_price = price + self.slippage
                    profit = (self.position.entry_price - exit_price) * self.unit
                self.realized_pnl += profit
                reward += profit
                profit_record = profit
                self.position.close()
            else:
                profit_record = 0.0

            # per-tick unrealized shaping
            if self.position.is_open():
                if self.position.side == 'long':
                    unreal = (price - self.position.entry_price) * self.unit
                else:
                    unreal = (self.position.entry_price - price) * self.unit
                reward += 0.05 * unreal

            done = False  # episode ends at end of day; we will mark done at final row
            # store transition for PPO
            trans = Transition(obs=obs, action=action, logp=logp, reward=reward, done=done, value=value)
            self.transitions.append(trans)
            # record transaction row
            df_transactions.append({'Time': t, 'Price': price, 'Action': ActionMap[action], 'Profit': profit_record})

            # Periodically run PPO update to stabilize memory usage
            if len(self.transitions) >= ppo_update_after_ticks:
                self.agent.ppo_update(self.transitions, gamma=self.gamma)
                # clear transitions
                self.transitions = []

        # End of day: force repay if position open
        if self.position.is_open():
            # use last price to force close
            last_price = float(df.iloc[-1]['Price'])
            if self.position.side == 'long':
                exit_price = last_price - self.slippage
                profit = (exit_price - self.position.entry_price) * self.unit
            else:
                exit_price = last_price + self.slippage
                profit = (self.position.entry_price - exit_price) * self.unit
            self.realized_pnl += profit
            # record forced REPAY as last transaction (append to df_transactions)
            df_transactions.append(
                {'Time': float(df.iloc[-1]['Time']), 'Price': last_price, 'Action': "REPAY", 'Profit': profit})
            # add final transition reward (closing)
            if self.transitions is not None:
                # append a dummy transition for the forced repay
                obs = self._compute_features(last_price, float(df.iloc[-1]['Volume']))
                if obs is not None:
                    trans = Transition(obs=obs, action=3, logp=0.0, reward=profit, done=True, value=0.0)
                    self.transitions.append(trans)
            self.position.close()

        # Final PPO update with remaining transitions
        if len(self.transitions) > 0:
            # mark last transition as done
            # set done=True for final
            last_idx = len(self.transitions) - 1
            last_trans = self.transitions[last_idx]
            self.transitions[last_idx] = Transition(obs=last_trans.obs, action=last_trans.action, logp=last_trans.logp,
                                                    reward=last_trans.reward, done=True, value=last_trans.value)
            self.agent.ppo_update(self.transitions, gamma=self.gamma)
            self.transitions = []

        # Save models
        self.agent.save(self.model_path, self.value_path)

        df_transaction = pd.DataFrame(df_transactions)
        # ensure Action/Hold for every Time as required by spec already recorded
        return df_transaction


# If this file is executed, provide a small self-test with synthetic data
if __name__ == '__main__':
    # minimal smoke test: synthetic random walk ticks for a day (~500 ticks)
    import time

    np.random.seed(0)
    N = 800
    base_price = 1000.0
    times = np.arange(N).astype(float)
    price = base_price + np.cumsum(np.random.randn(N) * 0.5)
    cum_vol = np.cumsum(np.random.poisson(10, size=N))
    df = pd.DataFrame({'Time': times, 'Price': price, 'Volume': cum_vol})

    trainer = Trainer(model_path='policy.pth', value_path='value.pth', device='cpu')
    print("Starting training (smoke test) ...")
    df_trans = trainer.train(df, epochs=1, ppo_update_after_ticks=200)
    print("Training finished. Transactions sample:")
    print(df_trans.tail(10))

    # Inference smoke
    sim = TradingSimulator(model_path='policy.pth', device='cpu', epsilon=0.01)
    print("Starting inference (smoke test) ...")
    actions = []
    for i in range(N):
        a = sim.add(float(times[i]), float(price[i]), float(cum_vol[i]))
        actions.append(a)
    print("Inference sample actions:", actions[-20:])
