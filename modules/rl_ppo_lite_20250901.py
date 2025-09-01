"""
RL Day-Trade Simulation Sample
- Inference on low-power CPU (Intel N150 on Windows 11): TradingSimulation
- Offline training on CPU (Ryzen 7 5700G on Fedora): Trainer

Environment (tested targets from your spec):
  Python 3.13+
  numpy==2.3.2, pandas==2.3.2, torch==2.8.0, gymnasium==1.2.0 (not directly used)

Interface:
  TradingSimulation.add(ts: float, price: float, volume: float, force_close: bool=False) -> str
  Trainer.train(df: pd.DataFrame) -> float

Notes:
- Action space: 0 HOLD, 1 BUY, 2 SELL, 3 REPAY (externally maps to strings)
- Single position only (long or short), lot=100, tick=1 JPY, slippage = 1 tick
- Realized PnL adds to reward at repayment; a fraction of unrealized PnL is shaped per tick
- Automatic warmup gate: until enough samples for indicators (n=60), only HOLD
- Models are small MLPs, TorchScript-compatible, saved to `policy.pth` by default
- Deterministic greedy action at inference; training uses categorical sampling
- Threading tuned for low-latency CPU inference
"""

from __future__ import annotations
import math
import os
import sys
from dataclasses import dataclass
from collections import deque
from typing import Deque, Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# ====== Constants / Trading Rules ======
LOT_SIZE = 100
TICK = 1.0  # JPY
SLIPPAGE = 1.0  # 1 tick
FEATURE_N = 60  # lookback for indicators
UNREALIZED_SHAPING = 0.05  # 5% of unrealized PnL per tick as shaping reward
POLICY_PATH = "policy.pth"
DEVICE = torch.device("cpu")

# Reduce CPU contention for low-power devices
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

ACTION_TO_STR = {0: "HOLD", 1: "BUY", 2: "SELL", 3: "REPAY"}


# ====== Feature Builder (online) ======
@dataclass
class OnlineStats:
    n: int = 0
    mean: float = 0.0
    M2: float = 0.0  # sum of squares of differences from the current mean (Welford)

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def var(self) -> float:
        return self.M2 / (self.n - 1) if self.n > 1 else 0.0


class FeatureBuilder:
    """Maintains rolling indicators with O(1) updates for live ticks."""

    def __init__(self, n: int = FEATURE_N):
        self.n = n
        self.prices: Deque[float] = deque(maxlen=n)
        self.returns: Deque[float] = deque(maxlen=n)
        self.avg_gain: Optional[float] = None
        self.avg_loss: Optional[float] = None
        self.running = OnlineStats()
        self.last_price: Optional[float] = None
        self.last_volume: Optional[float] = None

    def _update_rsi(self, price: float) -> float:
        if self.last_price is None:
            self.last_price = price
            return 50.0
        change = price - self.last_price
        self.last_price = price
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        alpha = 1.0 / self.n
        if self.avg_gain is None:
            self.avg_gain = gain
            self.avg_loss = loss
        else:
            self.avg_gain = (1 - alpha) * self.avg_gain + alpha * gain
            self.avg_loss = (1 - alpha) * self.avg_loss + alpha * loss
        rs = (self.avg_gain / (self.avg_loss + 1e-8)) if self.avg_loss is not None else 0.0
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return float(rsi)

    def update(self, price: float, volume_cum: float) -> Optional[np.ndarray]:
        # Δvolume (non-negative), log1p(Δvolume)
        if self.last_volume is None:
            dvol = 0.0
        else:
            raw = volume_cum - self.last_volume
            dvol = raw if raw > 0 else 0.0
        self.last_volume = volume_cum
        log_dvol = math.log1p(dvol)

        # running stats for price_z
        self.running.update(price)
        price_z = (price - self.running.mean) / math.sqrt(self.running.var + 1e-8)

        # rolling MA and std
        self.prices.append(price)
        if len(self.prices) >= 2:
            self.returns.append(self.prices[-1] - self.prices[-2])
        ma = float(np.mean(self.prices)) if len(self.prices) >= 1 else float(price)
        vol = float(np.std(self.prices)) if len(self.prices) >= 2 else 0.0

        rsi = self._update_rsi(price)

        if len(self.prices) < self.n:
            return None  # warmup not complete

        feat = np.array([
            price,
            dvol,
            log_dvol,
            ma,
            vol,
            rsi,
            price_z,
        ], dtype=np.float32)
        return feat


# ====== Trading State & PnL ======
@dataclass
class Position:
    side: int = 0  # 0: flat, +1: long, -1: short
    entry: float = 0.0

    def can_buy(self) -> bool:
        return self.side == 0

    def can_sell(self) -> bool:
        return self.side == 0

    def can_repay(self) -> bool:
        return self.side != 0

    def unrealized(self, mid: float) -> float:
        if self.side == 0:
            return 0.0
        if self.side == +1:
            exit_price = mid - SLIPPAGE
            return (exit_price - self.entry) * LOT_SIZE
        else:  # short
            exit_price = mid + SLIPPAGE
            return (self.entry - exit_price) * LOT_SIZE

    def open_long(self, mid: float) -> None:
        self.side = +1
        self.entry = mid + SLIPPAGE

    def open_short(self, mid: float) -> None:
        self.side = -1
        self.entry = mid - SLIPPAGE

    def close(self, mid: float) -> float:
        if self.side == 0:
            return 0.0
        if self.side == +1:
            exit_price = mid - SLIPPAGE
            pnl = (exit_price - self.entry) * LOT_SIZE
        else:
            exit_price = mid + SLIPPAGE
            pnl = (self.entry - exit_price) * LOT_SIZE
        self.side = 0
        self.entry = 0.0
        return pnl


# ====== Policy Network ======
class PolicyNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, num_actions: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.logits = nn.Linear(hidden, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.logits(x)


# ====== Lightweight Inference Wrapper ======
class TradingSimulation:
    def __init__(self, policy_path: str = POLICY_PATH, feature_n: int = FEATURE_N):
        if not os.path.exists(policy_path):
            raise FileNotFoundError(f"Policy not found: {policy_path}")
        self.device = DEVICE
        self.fb = FeatureBuilder(n=feature_n)
        self.pos = Position()
        self._load_policy(policy_path)
        self.policy_path = policy_path

    def _load_policy(self, path: str):
        try:
            obj = torch.jit.load(path, map_location=self.device)
            self.policy = obj
        except Exception:
            # fallback: load state_dict
            state = torch.load(path, map_location=self.device)
            self.policy = PolicyNet(in_dim=7)
            self.policy.load_state_dict(state)
            self.policy.eval()
        self.policy.eval()
        for p in self.policy.parameters():
            p.requires_grad_(False)

    def _constrain(self, action: int) -> int:
        # Enforce trading rules automatically
        if action == 1 and not self.pos.can_buy():
            return 0
        if action == 2 and not self.pos.can_sell():
            return 0
        if action == 3 and not self.pos.can_repay():
            return 0
        return action

    def add(self, ts: float, price: float, volume: float, force_close: bool = False) -> str:
        # Feature update
        feat = self.fb.update(price, volume)
        if force_close and self.pos.can_repay():
            _ = self.pos.close(price)
            return ACTION_TO_STR[3]
        if feat is None:
            return ACTION_TO_STR[0]

        with torch.no_grad():
            x = torch.from_numpy(feat).to(self.device).unsqueeze(0)
            logits = self.policy(x)
            action = int(torch.argmax(logits, dim=1).item())
        action = self._constrain(action)

        # Simulate fills
        if action == 1:
            self.pos.open_long(price)
        elif action == 2:
            self.pos.open_short(price)
        elif action == 3:
            _ = self.pos.close(price)
        return ACTION_TO_STR[action]


# ====== Trainer ======
class Trainer:
    def __init__(self, policy_path: str = POLICY_PATH, feature_n: int = FEATURE_N):
        self.device = DEVICE
        self.fb = FeatureBuilder(n=feature_n)
        self.policy_path = policy_path
        self._model, self._newly_created = self._load_or_init_model()

    def _load_or_init_model(self) -> Tuple[nn.Module, bool]:
        if os.path.exists(self.policy_path):
            try:
                print("[Trainer] Loading existing scripted policy ...")
                model = torch.jit.load(self.policy_path, map_location=self.device)
                model.eval()
                return model, False
            except Exception:
                try:
                    print("[Trainer] Loading existing state_dict policy ...")
                    state = torch.load(self.policy_path, map_location=self.device)
                    model = PolicyNet(in_dim=7)
                    model.load_state_dict(state)
                    model.to(self.device)
                    return model, False
                except Exception:
                    print("[Trainer] Existing model invalid. Creating a new one and overwriting.")
        else:
            print("[Trainer] No existing model. Creating a new one.")
        model = PolicyNet(in_dim=7)
        model.to(self.device)
        return model, True

    # ===== Core backtest + trajectory builder =====
    def _rollout(self, df: pd.DataFrame) -> Tuple[List[np.ndarray], List[int], List[float], float]:
        pos = Position()
        fb = self.fb  # reuse
        fb.__init__(n=fb.n)  # reset state
        feats: List[np.ndarray] = []
        actions: List[int] = []
        rewards: List[float] = []
        realized_total = 0.0

        for i, row in df.iterrows():
            price = float(row["Price"])  # mid price
            volume_cum = float(row["Volume"])  # cumulative
            feat = fb.update(price, volume_cum)

            if feat is None:
                actions.append(0)
                feats.append(np.zeros(7, dtype=np.float32))
                rewards.append(0.0)
                continue

            # Sample action from current policy (exploration during training)
            with torch.no_grad():
                x = torch.from_numpy(feat).to(self.device).unsqueeze(0)
                logits = self._model(x)
                probs = F.softmax(logits, dim=-1)
                dist = Categorical(probs=probs)
                a = int(dist.sample().item())

            # Enforce constraints
            if a == 1 and not pos.can_buy():
                a = 0
            if a == 2 and not pos.can_sell():
                a = 0
            if a == 3 and not pos.can_repay():
                a = 0

            # Simulate fills + rewards
            r = 0.0
            if a == 1:
                pos.open_long(price)
            elif a == 2:
                pos.open_short(price)
            elif a == 3:
                pnl = pos.close(price)
                realized_total += pnl
                r += pnl  # realized PnL

            # Shaping: small fraction of unrealized PnL each tick
            r += UNREALIZED_SHAPING * pos.unrealized(price)

            feats.append(feat)
            actions.append(a)
            rewards.append(r)

        # Force close at end
        if pos.side != 0:
            pnl = pos.close(float(df.iloc[-1]["Price"]))
            realized_total += pnl
            rewards[-1] += pnl
        return feats, actions, rewards, realized_total

    def train(self, df: pd.DataFrame, epochs: int = 3, batch_size: int = 4096, lr: float = 1e-3,
              clip: float = 0.2) -> float:
        assert {"Time", "Price", "Volume"}.issubset(df.columns), "df must have Time, Price, Volume"
        if len(df) < FEATURE_N + 5:
            print("[Trainer] Not enough data for warmup; skipping training.")
            return 0.0

        # ensure sorted by time
        df = df.sort_values("Time").reset_index(drop=True)

        # (1) rollout under current policy
        feats, actions, rewards, realized_total = self._rollout(df)

        # (2) compute returns-to-go (simple) and advantages (baseline = moving average)
        R = []
        g = 0.0
        gamma = 0.999  # very weak discounting for 1s ticks
        for r in reversed(rewards):
            g = r + gamma * g
            R.append(g)
        R = list(reversed(R))
        R = np.array(R, dtype=np.float32)
        baseline = pd.Series(R).rolling(600, min_periods=1).mean().to_numpy(dtype=np.float32)
        adv = R - baseline
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)

        # (3) train PPO-lite (clipped policy gradient without value net)
        model = self._model
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        X = torch.from_numpy(np.stack(feats)).to(self.device)
        A = torch.from_numpy(np.array(actions, dtype=np.int64)).to(self.device)
        ADV = torch.from_numpy(adv).to(self.device)

        with torch.no_grad():
            logits_old = model(X)
            pi_old = F.softmax(logits_old, dim=-1)
            logp_old = torch.log(pi_old.gather(1, A.view(-1, 1)).squeeze(1) + 1e-8)

        N = X.shape[0]
        idx = np.arange(N)
        for ep in range(epochs):
            np.random.shuffle(idx)
            for start in range(0, N, batch_size):
                sl = idx[start:start + batch_size]
                xb = X[sl]
                ab = A[sl]
                advb = ADV[sl]
                logits = model(xb)
                pi = F.softmax(logits, dim=-1)
                logp = torch.log(pi.gather(1, ab.view(-1, 1)).squeeze(1) + 1e-8)
                ratio = torch.exp(logp - logp_old[sl])
                obj1 = ratio * advb
                obj2 = torch.clamp(ratio, 1 - clip, 1 + clip) * advb
                loss = -(torch.min(obj1, obj2)).mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        model.eval()

        # Save TorchScript for fast, dependency-light inference
        try:
            scripted = torch.jit.script(model)
            scripted.save(self.policy_path)
            print(f"[Trainer] Saved scripted model to {self.policy_path}")
        except Exception as e:
            print(f"[Trainer] TorchScript failed: {e}; saving state_dict fallback.")
            torch.save(model.state_dict(), self.policy_path)
            print(f"[Trainer] Saved state_dict to {self.policy_path}")

        return float(realized_total)


# ====== Example usage ======
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "live"], help="train: offline learning from CSV; live: stdin ticks")
    parser.add_argument("--csv", type=str, default=None,
                        help="CSV path with columns Time,Price,Volume (for train mode)")
    parser.add_argument("--policy", type=str, default=POLICY_PATH, help="Path to policy file")
    args = parser.parse_args()

    if args.mode == "train":
        assert args.csv is not None, "--csv required in train mode"
        df = pd.read_csv(args.csv)
        t = Trainer(policy_path=args.policy)
        pnl = t.train(df)
        print(f"[Trainer] Realized PnL (JPY): {pnl:.2f}")
    else:
        sim = TradingSimulation(policy_path=args.policy)
        print("ts,price,volume,action")
        try:
            for line in sys.stdin:
                ts, price, volume = map(float, line.strip().split(","))
                action = sim.add(ts, price, volume)
                print(f"{ts},{price},{volume},{action}")
        except KeyboardInterrupt:
            pass
