"""
trading_ppo_simulator.py

Python 3.13.7
Dependencies assumed:
- gymnasium==1.2.0
- numpy==2.3.2
- pandas==2.3.2
- torch==2.8.0

This single-file example implements:
- TradingSimulation class (for inference; uses saved policy.pth/value.pth)
- Trainer class (for training on 1-day tick DataFrame)
- A lightweight PPO implementation with separate policy and value networks
- Feature calculation per requirements (60-tick warmup, log1p of delta volume,
  MA60, STD60, RSI60, Z-score60)
- Reward: realized PnL on repay added; holding gives 5% of (unrealized PnL) each tick
- Slippage and trade rules as specified (100-share lot, single position, no pyramiding)
- Epsilon-greedy exploration during training

Notes / simplifications:
- Uses simple advantages (GAE lambda optional) and PPO clipped objective
- Trainer.train returns df_transaction (one row per time x action combination)
- At the end of day, if position open -> forcibly repay at last price
- TradingSimulation.add returns one of: "HOLD","BUY","SELL","REPAY"

Save / load model files:
- policy -> policy.pth
- value  -> value.pth

Usage:
- Training (PC2):
    from trading_ppo_simulator import Trainer
    trainer = Trainer(model_dir="./models", device="cpu")
    df = pd.read_csv("one_day_ticks.csv")  # with Time, Price, Volume (cumulative)
    df_tx = trainer.train(df)
    df_tx.to_csv("transactions.csv", index=False)

- Inference (PC1):
    from trading_ppo_simulator import TradingSimulation
    sim = TradingSimulation(model_dir="./models", device="cpu")
    action = sim.add(time, price, volume)

"""

from __future__ import annotations
import os
import math
import random
import copy
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------- Utilities / Features ----------------------------

def calc_features_from_buffer(price_buf: np.ndarray, volume_buf: np.ndarray) -> Optional[np.ndarray]:
    """Given buffers length >=60, compute features vector.
    price_buf: np.array of prices (last N, N>=60)
    volume_buf: np.array of cumulative volumes (last N, N>=60)

    Returns: feature vector (shape=(5,)) -> [dvol_log1p, ma60, std60, rsi60, zscore]
    """
    if len(price_buf) < 60 or len(volume_buf) < 60:
        return None

    # ΔVolume (per-tick): current cumulative - previous cumulative
    dvol = np.diff(volume_buf[-61:])  # length 60
    # ensure non-negative
    dvol = np.maximum(dvol, 0)
    dvol_log = np.log1p(dvol)  # shape (60,)
    dvol_log1 = dvol_log[-1]  # latest

    prices_60 = price_buf[-60:]
    ma60 = float(np.mean(prices_60))
    std60 = float(np.std(prices_60, ddof=0))

    # RSI calculation (n=60)
    deltas = np.diff(prices_60)
    up = deltas.clip(min=0)
    down = -deltas.clip(max=0)
    if up.size == 0:
        rsi = 50.0
    else:
        avg_gain = up.mean()
        avg_loss = down.mean()
        if avg_loss == 0 and avg_gain == 0:
            rsi = 50.0
        elif avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / (avg_loss + 1e-12)
            rsi = 100.0 - (100.0 / (1.0 + rs))

    # z-score for latest price relative to 60 window
    if std60 < 1e-12:
        zscore = 0.0
    else:
        zscore = (prices_60[-1] - ma60) / std60

    feats = np.array([dvol_log1, ma60, std60, rsi, zscore], dtype=np.float32)
    return feats

# ---------------------------- Neural Nets ----------------------------

class PolicyNet(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: List[int], out_dim: int = 4):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, out_dim))  # logits for 4 actions
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # logits

class ValueNet(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: List[int]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

# ---------------------------- PPO Agent ----------------------------

class PPOAgent:
    def __init__(self,
                 input_dim: int,
                 policy_hidden: List[int] = [128, 64],
                 value_hidden: List[int] = [128, 64],
                 lr_policy: float = 3e-4,
                 lr_value: float = 1e-3,
                 clip_eps: float = 0.2,
                 vf_coef: float = 0.5,
                 ent_coef: float = 0.01,
                 device: str = "cpu"):
        self.device = torch.device(device)
        self.policy = PolicyNet(input_dim, policy_hidden, out_dim=4).to(self.device)
        self.value = ValueNet(input_dim, value_hidden).to(self.device)
        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.optimizer_value = optim.Adam(self.value.parameters(), lr=lr_value)
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef

    def select_action(self, obs: np.ndarray, epsilon: float = 0.0) -> Tuple[int, float]:
        """Return action int and logprob. Epsilon-greedy applied on top of policy distribution.
        """
        x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
        logits = self.policy(x)
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]
        # epsilon-greedy: with prob epsilon pick uniform random
        if random.random() < epsilon:
            action = int(np.random.choice(len(probs)))
            logp = math.log(max(probs[action], 1e-12))
            return action, logp
        else:
            action = int(np.random.choice(len(probs), p=probs))
            logp = math.log(max(probs[action], 1e-12))
            return action, logp

    def save(self, path_dir: str):
        os.makedirs(path_dir, exist_ok=True)
        torch.save(self.policy.state_dict(), os.path.join(path_dir, "policy.pth"))
        torch.save(self.value.state_dict(), os.path.join(path_dir, "value.pth"))

    def load(self, path_dir: str):
        pth_policy = os.path.join(path_dir, "policy.pth")
        pth_value = os.path.join(path_dir, "value.pth")
        if not os.path.exists(pth_policy) or not os.path.exists(pth_value):
            raise FileNotFoundError(f"Model files not found in {path_dir}")
        self.policy.load_state_dict(torch.load(pth_policy, map_location=self.device))
        self.value.load_state_dict(torch.load(pth_value, map_location=self.device))

    def update(self, batch: Dict[str, np.ndarray], epochs: int = 4, batch_size: int = 64):
        """PPO update given collected batch.
        batch contains: obs, actions, returns, advantages, old_logp
        All numpy arrays.
        """
        obs = torch.from_numpy(batch['obs']).to(self.device)
        actions = torch.from_numpy(batch['actions']).long().to(self.device)
        returns = torch.from_numpy(batch['returns']).to(self.device)
        advantages = torch.from_numpy(batch['advantages']).to(self.device)
        old_logp = torch.from_numpy(batch['old_logp']).to(self.device)

        N = obs.shape[0]
        inds = np.arange(N)
        for _ in range(epochs):
            np.random.shuffle(inds)
            for start in range(0, N, batch_size):
                mb_inds = inds[start:start+batch_size]
                mb_obs = obs[mb_inds]
                mb_actions = actions[mb_inds]
                mb_returns = returns[mb_inds]
                mb_adv = advantages[mb_inds]
                mb_old_logp = old_logp[mb_inds]

                logits = self.policy(mb_obs)
                logp_all = torch.log_softmax(logits, dim=-1)
                mb_logp = logp_all.gather(1, mb_actions.unsqueeze(1)).squeeze(1)
                ratio = torch.exp(mb_logp - mb_old_logp)

                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_adv
                policy_loss = -torch.mean(torch.min(surr1, surr2))

                # entropy
                probs = torch.softmax(logits, dim=-1)
                entropy = -torch.mean(torch.sum(probs * torch.log(probs + 1e-12), dim=1))

                # value loss
                value_pred = self.value(mb_obs)
                value_loss = torch.mean((mb_returns - value_pred) ** 2)

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                # update policy
                self.optimizer_policy.zero_grad()
                self.optimizer_value.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.value.parameters(), 0.5)
                self.optimizer_policy.step()
                self.optimizer_value.step()

# ---------------------------- Trading Simulator (Inference) ----------------------------

class TradingSimulation:
    """Inference-only simulator. Loads trained models from model_dir.

    Use add(time, price, volume) -> action_string
    """
    ACTION_MAP = {0: "HOLD", 1: "BUY", 2: "SELL", 3: "REPAY"}

    def __init__(self, model_dir: str = "./models", device: str = "cpu"):
        self.model_dir = model_dir
        self.device = device
        self.agent = PPOAgent(input_dim=8, device=device)  # 5 feats + position(1)+price_norm(1)+tick_since_open(1) approx
        # Check models exist
        self.agent.load(model_dir)

        # buffers
        self.price_buf: List[float] = []
        self.volume_buf: List[float] = []
        self.time_buf: List[float] = []

        # position state
        self.position = 0  # 0 none, 1 long, -1 short
        self.entry_price = 0.0
        self.lot = 100
        self.slippage = 1.0  # 1 tick as JPY
        self.warmup = 60
        self.ticks = 0

    def _build_obs(self) -> Optional[np.ndarray]:
        feats = calc_features_from_buffer(np.array(self.price_buf), np.array(self.volume_buf))
        if feats is None:
            return None
        # augment features: normalized price relative to MA, position flag, time_since_entry
        ma60 = feats[1]
        price_norm = 0.0 if ma60 == 0 else (self.price_buf[-1] - ma60) / (ma60 + 1e-12)
        pos_flag = float(self.position)  # -1,0,1
        ticks_since_entry = 0.0
        if self.position != 0 and hasattr(self, 'entry_tick_index'):
            ticks_since_entry = float(self.ticks - self.entry_tick_index)
        obs = np.concatenate([feats, np.array([price_norm, pos_flag, ticks_since_entry], dtype=np.float32)])
        return obs

    def add(self, time: float, price: float, volume: float) -> str:
        """Called every tick with cumulative volume.
        Returns action string.
        """
        # append
        self.time_buf.append(time)
        self.price_buf.append(price)
        self.volume_buf.append(volume)
        self.ticks += 1

        # warmup
        if len(self.price_buf) < self.warmup:
            return "HOLD"

        obs = self._build_obs()
        if obs is None:
            return "HOLD"

        # select action from policy
        action_idx, _ = self.agent.select_action(obs, epsilon=0.0)  # no exploration in inference
        # apply action constraints: after BUY/SELL only HOLD or REPAY allowed until REPAY
        if self.position != 0:
            # legal actions: HOLD(0) or REPAY(3)
            if action_idx not in (0, 3):
                action_idx = 0
        else:
            # if no position, cannot REPAY
            if action_idx == 3:
                action_idx = 0

        # apply action and update position state
        if action_idx == 1:  # BUY (open long)
            # entry price = market price + slippage
            self.position = 1
            self.entry_price = price + self.slippage
            self.entry_tick_index = self.ticks
            return "BUY"
        elif action_idx == 2:  # SELL (open short)
            self.position = -1
            self.entry_price = price - self.slippage
            self.entry_tick_index = self.ticks
            return "SELL"
        elif action_idx == 3:  # REPAY
            # close position
            if self.position == 1:
                exit_price = price - self.slippage
                pnl = (exit_price - self.entry_price) * self.lot
            elif self.position == -1:
                exit_price = price + self.slippage
                pnl = (self.entry_price - exit_price) * self.lot
            else:
                pnl = 0.0
            # reset
            self.position = 0
            self.entry_price = 0.0
            self.entry_tick_index = None
            return "REPAY"
        else:
            # HOLD
            return "HOLD"

# ---------------------------- Trainer ----------------------------

class TradingEnvSimulator:
    """A lightweight environment used by Trainer to simulate one day and gather transitions.

    Tracks position, calculates rewards, and logs transactions.
    """
    def __init__(self, agent: PPOAgent, slippage: float = 1.0, lot: int = 100):
        self.agent = agent
        self.slippage = slippage
        self.lot = lot
        self.warmup = 60

    def run_episode(self, df: pd.DataFrame, epsilon: float = 0.1) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
        """Run one pass over df and collect trajectory for PPO.
        Returns batch dict usable by agent.update and df_transaction.
        df expected to have columns Time, Price, Volume (cumulative)
        """
        price_buf = []
        volume_buf = []
        time_buf = []

        observations = []
        actions = []
        rewards = []
        old_logp = []
        values = []

        # df_transaction: record for every time and action
        rows = []

        position = 0
        entry_price = 0.0
        entry_tick_index = None

        ticks = 0

        for idx, row in df.iterrows():
            t = float(row['Time'])
            p = float(row['Price'])
            v = float(row['Volume'])
            price_buf.append(p)
            volume_buf.append(v)
            time_buf.append(t)
            ticks += 1

            # for transaction logging: we will append a row for every possible action
            # but for learning we store only the actually taken action

            # warmup
            if len(price_buf) < self.warmup:
                # log: only HOLD allowed
                # create entry for all actions per specification
                for act in ("HOLD", "BUY", "SELL", "REPAY"):
                    rows.append({'Time': t, 'Price': p, 'Action': act, 'Profit': 0.0})
                continue

            feats = calc_features_from_buffer(np.array(price_buf), np.array(volume_buf))
            ma60 = feats[1]
            price_norm = 0.0 if ma60 == 0 else (p - ma60) / (ma60 + 1e-12)
            pos_flag = float(position)
            ticks_since_entry = 0.0
            if position != 0 and entry_tick_index is not None:
                ticks_since_entry = float(ticks - entry_tick_index)
            obs = np.concatenate([feats, np.array([price_norm, pos_flag, ticks_since_entry], dtype=np.float32)])

            # policy selection with epsilon-greedy for exploration
            action_idx, logp = self.agent.select_action(obs, epsilon=epsilon)
            # enforce action constraints
            if position != 0:
                if action_idx not in (0, 3):
                    action_idx = 0
            else:
                if action_idx == 3:
                    action_idx = 0

            # compute immediate reward: if repay -> realized pnl; while holding add 5% of unrealized per tick
            immediate_reward = 0.0

            # action effects
            if action_idx == 1 and position == 0:
                position = 1
                entry_price = p + self.slippage
                entry_tick_index = ticks
            elif action_idx == 2 and position == 0:
                position = -1
                entry_price = p - self.slippage
                entry_tick_index = ticks
            elif action_idx == 3 and position != 0:
                # close
                if position == 1:
                    exit_price = p - self.slippage
                    pnl = (exit_price - entry_price) * self.lot
                else:
                    exit_price = p + self.slippage
                    pnl = (entry_price - exit_price) * self.lot
                immediate_reward += pnl
                # reset
                position = 0
                entry_price = 0.0
                entry_tick_index = None

            # holding reward: 5% of unrealized per tick
            if position != 0:
                if position == 1:
                    unreal = (p - entry_price) * self.lot
                else:
                    unreal = (entry_price - p) * self.lot
                immediate_reward += 0.05 * unreal

            # value estimate
            with torch.no_grad():
                val = float(self.agent.value(torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.agent.device)).cpu().numpy()[0])

            # append to trajectory
            observations.append(obs)
            actions.append(action_idx)
            rewards.append(immediate_reward)
            old_logp.append(logp)
            values.append(val)

            # Fill transaction log rows (all actions)
            # Profit only when action==REPAY and that action was legal and closed something
            # We'll store 0 for others (spec says record all actions)
            # For the actual chosen action record the realized profit if repay
            profit_for_log = 0.0
            if action_idx == 3:
                # immediate_reward already contains realized pnl (and 5% holding portion) - we want only realized
                # attempt to recompute realized pnl (safer): if last step closed, we computed pnl above
                # For logging, record pnl when repay happened (otherwise 0)
                # To avoid double counting, here set Profit = pnl
                profit_for_log = pnl if 'pnl' in locals() else 0.0
            rows.append({'Time': t, 'Price': p, 'Action': self._act_str(0), 'Profit': 0.0})
            rows.append({'Time': t, 'Price': p, 'Action': self._act_str(1), 'Profit': 0.0})
            rows.append({'Time': t, 'Price': p, 'Action': self._act_str(2), 'Profit': 0.0})
            rows.append({'Time': t, 'Price': p, 'Action': self._act_str(3), 'Profit': profit_for_log})

        # Force close if position still open at end
        if position != 0:
            p = float(df.iloc[-1]['Price'])
            if position == 1:
                exit_price = p - self.slippage
                pnl = (exit_price - entry_price) * self.lot
            else:
                exit_price = p + self.slippage
                pnl = (entry_price - exit_price) * self.lot
            # record as final repay step
            observations.append(observations[-1])
            actions.append(3)
            rewards.append(pnl)
            old_logp.append(0.0)
            values.append(0.0)
            rows.append({'Time': float(df.iloc[-1]['Time']), 'Price': p, 'Action': self._act_str(3), 'Profit': pnl})

        # compute returns and advantages (simple lambda-free GAE)
        obs_arr = np.array(observations, dtype=np.float32)
        actions_arr = np.array(actions, dtype=np.int32)
        old_logp_arr = np.array(old_logp, dtype=np.float32)
        rewards_arr = np.array(rewards, dtype=np.float32)
        values_arr = np.array(values, dtype=np.float32)

        # compute discounted returns
        gamma = 0.99
        returns = np.zeros_like(rewards_arr)
        running = 0.0
        for t in reversed(range(len(rewards_arr))):
            running = rewards_arr[t] + gamma * running
            returns[t] = running

        advantages = returns - values_arr

        batch = {
            'obs': obs_arr,
            'actions': actions_arr,
            'returns': returns.astype(np.float32),
            'advantages': advantages.astype(np.float32),
            'old_logp': old_logp_arr
        }

        df_tx = pd.DataFrame(rows)
        return batch, df_tx

    @staticmethod
    def _act_str(i: int) -> str:
        return {0: 'HOLD', 1: 'BUY', 2: 'SELL', 3: 'REPAY'}[i]

class Trainer:
    def __init__(self, model_dir: str = './models', device: str = 'cpu'):
        self.model_dir = model_dir
        self.device = device
        self.agent = PPOAgent(input_dim=8, device=device)
        self.env_sim = TradingEnvSimulator(self.agent)

    def train(self, df: pd.DataFrame,
              epochs: int = 10,
              steps_per_epoch: int = 4,
              epsilon_start: float = 0.2,
              epsilon_end: float = 0.01,
              save_every: int = 1) -> pd.DataFrame:
        """Train on one-day df. Returns df_transaction (rows x 4 actions recorded per tick).

        We will run a few epochs where in each epoch we run steps_per_epoch episodes (passes over df)
        and perform PPO updates using the batches collected.
        """
        all_dfs = []
        collected_rows = []

        for ep in range(epochs):
            eps = epsilon_start - (epsilon_start - epsilon_end) * (ep / max(1, epochs-1))
            # collect multiple rollouts
            batches = {'obs': [], 'actions': [], 'returns': [], 'advantages': [], 'old_logp': []}
            df_tx_list = []
            for s in range(steps_per_epoch):
                batch, df_tx = self.env_sim.run_episode(df, epsilon=eps)
                for k in batches.keys():
                    batches[k].append(batch[k])
                df_tx_list.append(df_tx)

            # concat
            for k in list(batches.keys()):
                batches[k] = np.concatenate(batches[k], axis=0)

            # normalize advantages
            adv = batches['advantages']
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            batches['advantages'] = adv

            # perform PPO update
            self.agent.update(batches, epochs=4, batch_size=64)

            # optionally save
            if (ep + 1) % save_every == 0:
                os.makedirs(self.model_dir, exist_ok=True)
                self.agent.save(self.model_dir)

            # collect transaction logs
            collected_rows.extend(pd.concat(df_tx_list, ignore_index=True).to_dict('records'))

        df_transaction = pd.DataFrame(collected_rows)
        # ensure final forced repay handled in env
        return df_transaction

# ---------------------------- Helper / Main (example) ----------------------------

if __name__ == '__main__':
    # small example of how to use Trainer with fake data
    # NOTE: Replace with real tick CSV for actual training
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()

    if args.demo:
        # build fake tick df (1 sec ticks, 200 ticks)
        N = 500
        ts = np.arange(N).astype(float)
        price = 1000 + np.cumsum(np.random.randn(N).astype(float))
        volume = np.cumsum(np.random.randint(1, 500, size=N))
        df = pd.DataFrame({'Time': ts, 'Price': price, 'Volume': volume})

        trainer = Trainer(model_dir='./models_demo', device='cpu')
        df_tx = trainer.train(df, epochs=3, steps_per_epoch=2)
        print('Transaction log sample:')
        print(df_tx.head())

        # show saved models
        print('Saved model files:', os.listdir('./models_demo'))

    else:
        print('This module provides Trainer and TradingSimulation classes. Use --demo to run a short demo.')
