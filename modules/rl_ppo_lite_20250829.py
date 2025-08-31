"""
ppo_lite_trading_sample.py

CPUオンリー（Windows 11 / Fedora Linux）で、ザラ場は軽量推論、引け後は学習する
継続学習ループ向けの最小実装サンプルです。

依存:
  - Python 3.13.7
  - torch==2.8.0
  - pandas==2.3.2
  - numpy==2.2.6
  - gymnasium==1.2.0 (未使用でもOK。依存はありません)

公開インターフェイス:
  - class TradingSimulation:
      add(ts: float, price: float, volume: float, force_close: bool=False) -> str
      finalize() -> pandas.DataFrame
  - class Trainer:
      train(df: pd.DataFrame) -> None

ファイル保存:
  - 既定のモデルファイルは "policy.pch"（PyTorch state_dict）

売買アクション（内部整数 -> 外部文字列）:
  0: HOLD, 1: BUY, 2: SELL, 3: REPAY

売買ルール（簡易約定）:
  - 売買単位=100株, 呼び値=1円, スリッページ=呼び値1倍
  - 同時保有は1建玉のみ（ロングまたはショート）
  - entry/exit はスリッページ考慮
  - 手数料は未考慮

特徴量（n=60）:
  - Δvolume = max(volume - last_volume, 0)
  - log1p(Δvolume)
  - 移動平均 MA(n)
  - ボラティリティ std(n)
  - RSI(n)
  - 価格正規化 price_z（Welford法のランニング平均/分散）
  - ウォームアップ不足や欠損時は学習せず/推論は HOLD

報酬:
  - 返済時の確定益（円）を即時報酬に加算
  - 含み益の一部（shaping_weight * unrealized_pnl）を各ティックで付与

注意:
  - 学習は on-policy（PPO-lite）で1日分のティックからロールアウトを構築
  - ザラ場の TradingSimulation は推論のみで軽量化（no_grad, fp32, 小型ネット）
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from typing import Optional, List, Tuple, Dict
import math
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# ==========================
# ユーティリティ
# ==========================
ACTION_STR = {0: "HOLD", 1: "BUY", 2: "SELL", 3: "REPAY"}
A_HOLD, A_BUY, A_SELL, A_REPAY = 0, 1, 2, 3


@dataclass
class TradeState:
    position: int = 0  # 0: flat, +1: long, -1: short
    entry_price: float = 0.0


# ==========================
# 特徴量抽出（オンライン対応）
# ==========================
class FeatureExtractor:
    def __init__(self, n: int = 60, eps: float = 1e-6):
        self.n = n
        self.eps = eps
        self.prices = deque(maxlen=n)
        self.price_diffs = deque(maxlen=n)
        self.last_volume: Optional[float] = None
        # 価格のランニング統計（Welford）
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0

    def _update_running_stats(self, x: float):
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def _running_std(self) -> float:
        if self.count < 2:
            return 0.0
        return math.sqrt(max(self.M2 / (self.count - 1), 0.0))

    def _compute_rsi(self) -> Optional[float]:
        # シンプルなRSI: n期間の平均上昇/下落から算出
        if len(self.price_diffs) < self.n:
            return None
        gains = [max(d, 0.0) for d in self.price_diffs]
        losses = [max(-d, 0.0) for d in self.price_diffs]
        avg_gain = (sum(gains) / self.n) if self.n > 0 else 0.0
        avg_loss = (sum(losses) / self.n) if self.n > 0 else 0.0
        if avg_loss <= self.eps:
            return 100.0
        rs = avg_gain / (avg_loss + self.eps)
        rsi = 100.0 * (1.0 - (1.0 / (1.0 + rs)))
        return rsi

    def update_and_get(self, price: float, volume: float) -> Optional[np.ndarray]:
        # Δvolume と log1p
        if self.last_volume is None:
            dvol = 0.0
        else:
            dvol = max(volume - self.last_volume, 0.0)
        self.last_volume = volume
        log_dvol = math.log1p(dvol)

        # 価格系列の更新
        if len(self.prices) > 0:
            self.price_diffs.append(price - self.prices[-1])
        self.prices.append(price)
        self._update_running_stats(price)

        # ウォームアップ不足
        if len(self.prices) < self.n:
            return None

        # MA と std（窓）
        arr = np.fromiter(self.prices, dtype=np.float64)
        ma = float(arr.mean())
        std = float(arr.std(ddof=0))

        # RSI
        rsi = self._compute_rsi()
        if rsi is None:
            return None

        # price_z
        run_std = self._running_std()
        price_z = (price - self.mean) / (run_std + self.eps)

        # 数値安定化のためのスケーリング
        rsi01 = rsi / 100.0
        std_s = math.log1p(std)
        ma_dev = (price - ma) / (std + self.eps)  # ボリンジャー的偏差

        feat = np.array([
            price_z,
            rsi01,
            std_s,
            ma_dev,
            log_dvol,
        ], dtype=np.float32)
        return feat


# ==========================
# PPO-lite Policy（Actor-Critic）
# ==========================
class PPOPolicy(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 64, act_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, act_dim)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value


# ==========================
# PPO-lite エージェント
# ==========================
@dataclass
class PPOConfig:
    obs_dim: int = 5
    act_dim: int = 4
    hidden: int = 64
    lr: float = 3e-4
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    train_iters: int = 4
    batch_size: int = 1024
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 1.0
    device: str = "cpu"


class PPOLiteAgent:
    def __init__(self, cfg: PPOConfig):
        self.cfg = cfg
        self.policy = PPOPolicy(cfg.obs_dim, cfg.hidden, cfg.act_dim).to(cfg.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[int, float, float]:
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.cfg.device).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.policy(x)
            if deterministic:
                action = int(logits.argmax(dim=-1).item())
                logp = F.log_softmax(logits, dim=-1)[0, action].item()
            else:
                dist = Categorical(logits=logits)
                a = dist.sample()
                action = int(a.item())
                logp = float(dist.log_prob(a).item())
            v = float(value.item())
        return action, logp, v

    def evaluate_actions(self, obs_b: torch.Tensor, act_b: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.policy(obs_b)
        dist = Categorical(logits=logits)
        logp = dist.log_prob(act_b)
        entropy = dist.entropy()
        return logp, entropy, values

    def update(self, rollouts: Dict[str, np.ndarray]):
        cfg = self.cfg
        obs = torch.as_tensor(rollouts["obs"], dtype=torch.float32, device=cfg.device)
        acts = torch.as_tensor(rollouts["acts"], dtype=torch.long, device=cfg.device)
        logp_old = torch.as_tensor(rollouts["logp"], dtype=torch.float32, device=cfg.device)
        returns = torch.as_tensor(rollouts["returns"], dtype=torch.float32, device=cfg.device)
        advantages = torch.as_tensor(rollouts["advantages"], dtype=torch.float32, device=cfg.device)

        # 正規化
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        n = obs.size(0)
        idx = np.arange(n)
        for _ in range(cfg.train_iters):
            np.random.shuffle(idx)
            for start in range(0, n, cfg.batch_size):
                end = min(start + cfg.batch_size, n)
                mb_idx = idx[start:end]
                mb_obs = obs[mb_idx]
                mb_acts = acts[mb_idx]
                mb_logp_old = logp_old[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advs = advantages[mb_idx]

                new_logp, entropy, values = self.evaluate_actions(mb_obs, mb_acts)
                ratio = (new_logp - mb_logp_old).exp()
                surr1 = ratio * mb_advs
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * mb_advs
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, mb_returns)
                entropy_loss = -entropy.mean()

                loss = policy_loss + cfg.vf_coef * value_loss + cfg.ent_coef * entropy_loss
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()


# ==========================
# 約定とPnL
# ==========================
@dataclass
class MarketConfig:
    lot_size: int = 100
    tick_size: float = 1.0  # 呼び値
    slippage_ticks: int = 1  # 呼び値の何倍か


class ExecutionEngine:
    def __init__(self, mcfg: MarketConfig):
        self.cfg = mcfg
        self.state = TradeState()

    def allowed_actions(self) -> List[int]:
        if self.state.position == 0:
            return [A_HOLD, A_BUY, A_SELL]
        else:
            return [A_HOLD, A_REPAY]

    def _entry_with_slip(self, price: float, side: int) -> float:
        slip = self.cfg.tick_size * self.cfg.slippage_ticks
        if side == +1:  # 新規買い
            return price + slip
        elif side == -1:  # 新規売り
            return price - slip
        else:
            return price

    def _exit_with_slip(self, price: float, side: int) -> float:
        slip = self.cfg.tick_size * self.cfg.slippage_ticks
        if side == +1:  # ロング解消
            return price - slip
        elif side == -1:  # ショート解消
            return price + slip
        else:
            return price

    def step(self, price: float, action: int) -> float:
        """実行し確定損益を返す（返済時のみ非ゼロ）。"""
        profit = 0.0
        ls = self.cfg.lot_size
        st = self.state

        # 非許可アクションは HOLD に置換
        if action not in self.allowed_actions():
            action = A_HOLD

        if st.position == 0:
            if action == A_BUY:
                st.position = +1
                st.entry_price = self._entry_with_slip(price, +1)
            elif action == A_SELL:
                st.position = -1
                st.entry_price = self._entry_with_slip(price, -1)
            # HOLD/REPAY -> 何もしない（REPAY は置換済）
        else:
            if action == A_REPAY:
                exit_price = self._exit_with_slip(price, st.position)
                if st.position == +1:
                    profit = (exit_price - st.entry_price) * ls
                else:
                    profit = (st.entry_price - exit_price) * ls
                st.position = 0
                st.entry_price = 0.0
            # HOLD なら何もしない
        return profit

    def unrealized_pnl(self, price: float) -> float:
        st = self.state
        if st.position == 0:
            return 0.0
        # 返済時のスリッページを見込んだ含み益
        exit_price = self._exit_with_slip(price, st.position)
        if st.position == +1:
            return (exit_price - st.entry_price) * self.cfg.lot_size
        else:
            return (st.entry_price - exit_price) * self.cfg.lot_size


# ==========================
# TradingSimulation（ザラ場: 推論のみ）
# ==========================
class TradingSimulation:
    def __init__(self, model_path: str = "policy.pch", feature_n: int = 60, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self.feat = FeatureExtractor(n=feature_n)
        self.market = ExecutionEngine(MarketConfig())
        self.cfg = PPOConfig(device=device)
        self.agent = PPOLiteAgent(self.cfg)
        # モデルのロード（存在しない場合はエラー）
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        state = torch.load(self.model_path, map_location=device)
        self.agent.policy.load_state_dict(state)
        self.agent.policy.eval()
        # ログ
        self.rows: List[Dict] = []

    def add(self, ts: float, price: float, volume: float, force_close: bool = False) -> str:
        # 特徴量
        feat = self.feat.update_and_get(price, volume)
        action = A_HOLD
        profit = 0.0

        if force_close and self.market.state.position != 0:
            action = A_REPAY
        elif feat is not None:
            # 軽量推論（deterministic）
            with torch.no_grad():
                a, _, _ = self.agent.act(feat, deterministic=True)
            # 非許可は内部で置換される（約定側）
            action = a
        else:
            action = A_HOLD

        profit = self.market.step(price, action)
        self.rows.append({
            "Time": float(ts),
            "Price": float(price),
            "Action": ACTION_STR[action],
            "profit": float(profit),
        })
        return ACTION_STR[action]

    def finalize(self) -> pd.DataFrame:
        df = pd.DataFrame(self.rows)
        self.rows.clear()
        return df


# ==========================
# Trainer（引け後: 再学習）
# ==========================
class Trainer:
    def __init__(self, model_path: str = "policy.pch", feature_n: int = 60, device: str = "cpu",
                 shaping_weight: float = 0.02):
        self.model_path = model_path
        self.device = device
        self.feat = FeatureExtractor(n=feature_n)
        self.market = ExecutionEngine(MarketConfig())
        self.cfg = PPOConfig(device=device)
        self.agent = PPOLiteAgent(self.cfg)
        self.shaping_weight = shaping_weight
        self._load_or_init_model()

    def _load_or_init_model(self):
        if os.path.exists(self.model_path):
            try:
                state = torch.load(self.model_path, map_location=self.device)
                self.agent.policy.load_state_dict(state)
                print("[Trainer] 既存モデルを読み込みました -> 継続学習します.")
                return
            except Exception as e:
                print(f"[Trainer] 既存モデルの読み込みに失敗 -> 新規作成します. reason={e}")
        else:
            print("[Trainer] 既存モデルなし -> 新規作成します.")
        # 新規初期化済み（__init__で）

    def _build_rollout(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        obs_list, act_list, logp_list, val_list, rew_list = [], [], [], [], []
        # 日中のロールアウト構築
        for i, row in df.iterrows():
            ts = float(row["Time"])  # 未使用（ログや将来拡張向け）
            price = float(row["Price"])
            volume = float(row["Volume"])  # 累計

            feat = self.feat.update_and_get(price, volume)
            # ウォームアップ中は HOLD & 報酬ゼロ（含み益も無し）
            if feat is None:
                # shaping のためだけに観測を欠損させない実装もあり得るが、簡易化
                continue

            # サンプリング行動
            a, logp, v = self.agent.act(feat, deterministic=False)

            # 強制クローズ（最終行で行う予定だが、ここでは通常の流れ）
            reward = 0.0

            # 実行して確定益
            realized = self.market.step(price, a)
            reward += realized

            # 含み益 shaping
            unreal = self.market.unrealized_pnl(price)
            reward += self.shaping_weight * unreal

            obs_list.append(feat)
            act_list.append(a)
            logp_list.append(logp)
            val_list.append(v)
            rew_list.append(reward)

        # 終端で強制返済
        if self.market.state.position != 0 and len(df) > 0:
            last_price = float(df.iloc[-1]["Price"])
            # 返済アクションを仮想的に追加
            feat = self.feat.update_and_get(last_price, float(df.iloc[-1]["Volume"]))
            if feat is not None:
                realized = self.market.step(last_price, A_REPAY)
                reward = realized  # shaping は無し
                a = A_REPAY
                with torch.no_grad():
                    x = torch.as_tensor(feat, dtype=torch.float32).unsqueeze(0)
                    _, v = self.agent.policy(x)
                    v = float(v.item())
                obs_list.append(feat)
                act_list.append(a)
                logp_list.append(0.0)  # 事後的な返済なので logp は0で良い（影響小）
                val_list.append(v)
                rew_list.append(reward)

        if len(obs_list) == 0:
            return {"obs": np.zeros((0, self.cfg.obs_dim), dtype=np.float32),
                    "acts": np.zeros((0,), dtype=np.int64),
                    "logp": np.zeros((0,), dtype=np.float32),
                    "values": np.zeros((0,), dtype=np.float32),
                    "rewards": np.zeros((0,), dtype=np.float32)}

        rollouts = {
            "obs": np.asarray(obs_list, dtype=np.float32),
            "acts": np.asarray(act_list, dtype=np.int64),
            "logp": np.asarray(logp_list, dtype=np.float32),
            "values": np.asarray(val_list, dtype=np.float32),
            "rewards": np.asarray(rew_list, dtype=np.float32),
        }
        return rollouts

    @staticmethod
    def _gae(rewards: np.ndarray, values: np.ndarray, gamma: float, lam: float) -> Tuple[np.ndarray, np.ndarray]:
        n = len(rewards)
        adv = np.zeros(n, dtype=np.float32)
        lastgaelam = 0.0
        for t in reversed(range(n)):
            next_value = values[t + 1] if t + 1 < n else 0.0
            delta = rewards[t] + gamma * next_value - values[t]
            lastgaelam = delta + gamma * lam * lastgaelam
            adv[t] = lastgaelam
        returns = values + adv
        return adv, returns

    def train(self, df: pd.DataFrame):
        """当日（および過去）ティックデータ df を使って学習し、モデルを上書き保存。"""
        required_cols = {"Time", "Price", "Volume"}
        if not required_cols.issubset(set(df.columns)):
            missing = required_cols - set(df.columns)
            raise ValueError(f"Trainer.train: df に必要な列が不足しています: {missing}")

        rollouts = self._build_rollout(df)
        if rollouts["obs"].shape[0] == 0:
            print("[Trainer] 有効なロールアウトがありません（ウォームアップ不足など）。学習をスキップします。")
            return

        # 価値推定（values は rollouts['values'] を利用）
        adv, ret = self._gae(
            rollouts["rewards"], rollouts["values"],
            self.cfg.gamma, self.cfg.gae_lambda
        )
        rollouts["advantages"] = adv
        rollouts["returns"] = ret

        # 学習
        self.agent.update(rollouts)

        # 保存
        torch.save(self.agent.policy.state_dict(), self.model_path)
        print(f"[Trainer] モデルを保存しました -> {self.model_path}")


# ==========================
# 参考: 簡易動作テスト（任意）
# ==========================
if __name__ == "__main__":
    # ダミーデータで学習 -> 推論の流れを確認
    rng = np.random.default_rng(0)
    t = np.arange(0, 3600, 1, dtype=float)  # 1時間ぶん
    price = 4000 + np.cumsum(rng.normal(0, 2, size=t.size))
    volume = np.cumsum(rng.integers(0, 1000, size=t.size).astype(float))

    df = pd.DataFrame({"Time": t, "Price": price, "Volume": volume})

    trainer = Trainer(model_path="policy.pch", feature_n=60, device="cpu")
    trainer.train(df)

    sim = TradingSimulation(model_path="policy.pch", feature_n=60, device="cpu")
    for i, row in df.iloc[:100].iterrows():
        act = sim.add(float(row["Time"]), float(row["Price"]), float(row["Volume"]))
    res = sim.finalize()
    print(res.tail())
