# ppo_trader.py
# Python 3.13 / CPU 前提
# torch==2.8.0, numpy==2.3.2, pandas==2.3.2, gymnasium==1.2.0(※未使用) を想定

import os
import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# =========================================================
# 仕様・定数
# =========================================================
ACTION_HOLD = 0
ACTION_BUY = 1
ACTION_SELL = 2
ACTION_REPAY = 3

ACTION_MAP = {
    ACTION_HOLD: "HOLD",
    ACTION_BUY: "BUY",
    ACTION_SELL: "SELL",
    ACTION_REPAY: "REPAY",
}

WARMUP = 60             # 特徴量用ウォームアップ
UNIT = 100              # 売買単位（株）
SLIPPAGE = 1.0          # 常に 1 ティック
EPSILON = 0.05          # ε-greedy（探索）
FEATURE_DIM = 5         # LogΔVol, MA, STD, RSI, ZScore

# PPO ハイパラ（適宜調整可）
GAMMA = 0.99
LAMBDA = 0.95           # GAE
CLIP_RANGE = 0.2
ENTROPY_COEF = 0.005
VALUE_COEF = 0.5
LR_POLICY = 3e-4
LR_VALUE = 1e-3
EPOCHS = 8              # PPO 更新エポック
MINIBATCH_SIZE = 2048   # ミニバッチサイズ（1日 19,500 ティック想定）


# =========================================================
# ユーティリティ（特徴量）
# =========================================================
def compute_rsi_series(price_series: pd.Series, n: int = 60) -> pd.Series:
    delta = price_series.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    # 単純移動平均ベース（EMAに変えるのも可）
    roll_up = up.rolling(n, min_periods=n).mean()
    roll_down = down.rolling(n, min_periods=n).mean()
    rs = roll_up / (roll_down + 1e-9)
    rsi = 100 - 100 / (1 + rs)
    return rsi


def compute_features_df(df: pd.DataFrame) -> pd.DataFrame:
    """ df: columns = [Time, Price, Volume(累計)] """
    df = df.copy()
    dvol = df["Volume"].diff().fillna(0)
    df["LogDeltaVolume"] = np.log1p(dvol.clip(lower=0))  # 成り行き等で負は0に丸める方針
    df["MA"] = df["Price"].rolling(WARMUP, min_periods=WARMUP).mean()
    df["STD"] = df["Price"].rolling(WARMUP, min_periods=WARMUP).std()
    df["RSI"] = compute_rsi_series(df["Price"], n=WARMUP)
    df["ZScore"] = (df["Price"] - df["MA"]) / (df["STD"] + 1e-9)
    return df


@dataclass
class FeatureScaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (self.std + 1e-9)

    @staticmethod
    def fit(features: np.ndarray) -> "FeatureScaler":
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        std[std < 1e-6] = 1.0  # ゼロ割り回避
        return FeatureScaler(mean=mean, std=std)


# =========================================================
# ニューラルネット
# =========================================================
class PolicyNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 4):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.head = nn.Linear(128, output_dim)  # ロジット（Softmaxは後段で分布計算時に使用）
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        logits = self.head(z)
        return logits


class ValueNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# =========================================================
# 行動マスク（売買制約）
# =========================================================
def action_mask(position: int) -> torch.Tensor:
    """
    position: 0=ノーポジ, +1=ロング, -1=ショート
    ルール:
      - ノーポジ: [HOLD, BUY, SELL] 可 / REPAY 不可
      - 保有中:   [HOLD, REPAY] 可 / BUY, SELL 不可
    """
    mask = torch.zeros(4, dtype=torch.bool)
    if position == 0:
        mask[ACTION_HOLD] = True
        mask[ACTION_BUY] = True
        mask[ACTION_SELL] = True
        mask[ACTION_REPAY] = False
    else:
        mask[ACTION_HOLD] = True
        mask[ACTION_REPAY] = True
        mask[ACTION_BUY] = False
        mask[ACTION_SELL] = False
    return mask


def masked_categorical(logits: torch.Tensor, mask: torch.Tensor) -> Categorical:
    """ 不可行動のロジットを -inf にして分布を作る """
    very_neg = torch.finfo(logits.dtype).min
    masked_logits = logits.clone()
    masked_logits[:, ~mask] = very_neg
    return Categorical(logits=masked_logits)


# =========================================================
# 取引ロジック（損益・ルール）
# =========================================================
def realize_pnl(position: int, entry_price: float, current_price: float) -> float:
    if position == 1:   # ロング解消
        return (current_price - SLIPPAGE - entry_price) * UNIT
    elif position == -1:  # ショート解消
        return (entry_price - (current_price + SLIPPAGE)) * UNIT
    return 0.0


def unrealized_pnl(position: int, entry_price: float, current_price: float) -> float:
    if position == 1:
        return (current_price - entry_price) * UNIT
    elif position == -1:
        return (entry_price - current_price) * UNIT
    return 0.0


# =========================================================
# PPO バッファ
# =========================================================
@dataclass
class Rollout:
    states: List[np.ndarray]
    actions: List[int]
    logprobs: List[float]
    rewards: List[float]
    dones: List[bool]
    values: List[float]


# =========================================================
# Trainer（学習専用：PC1）
# =========================================================
class Trainer:
    def __init__(self,
                 policy_path: str = "policy.pth",
                 value_path: str = "value.pth"):
        self.device = torch.device("cpu")
        self.policy = PolicyNet(FEATURE_DIM).to(self.device)
        self.value = ValueNet(FEATURE_DIM).to(self.device)
        self.optim_policy = optim.Adam(self.policy.parameters(), lr=LR_POLICY)
        self.optim_value = optim.Adam(self.value.parameters(), lr=LR_VALUE)

        self.policy_path = policy_path
        self.value_path = value_path
        self.scaler: FeatureScaler | None = None

    def _features_and_scaler(self, df: pd.DataFrame) -> Tuple[np.ndarray, FeatureScaler]:
        df_feat = compute_features_df(df)
        feats = df_feat[["LogDeltaVolume", "MA", "STD", "RSI", "ZScore"]].to_numpy(dtype=np.float32)
        # WARMUP 未満は0埋めだが、トレーニング中はHOLD強制なので特に問題なし
        # スケーラはウォームアップ以降でフィット（安定化）
        valid = np.arange(len(df)) >= WARMUP
        scaler = FeatureScaler.fit(feats[valid])
        feats_norm = scaler.transform(feats)
        return feats_norm, scaler

    def _ppo_update(self, data: Dict[str, torch.Tensor]) -> None:
        states = data["states"]
        actions = data["actions"]
        old_logprobs = data["logprobs"]
        returns = data["returns"]
        advantages = data["advantages"]

        n = states.size(0)
        idxs = np.arange(n)

        for _ in range(EPOCHS):
            np.random.shuffle(idxs)
            for start in range(0, n, MINIBATCH_SIZE):
                end = min(start + MINIBATCH_SIZE, n)
                mb_idx = torch.tensor(idxs[start:end], dtype=torch.long)

                s = states[mb_idx]
                a = actions[mb_idx]
                old_lp = old_logprobs[mb_idx]
                adv = advantages[mb_idx]
                ret = returns[mb_idx]

                # 新ロジット（注意：ここでは行動マスクを入れにくいので、logprobはサンプル時の分布に依存）
                logits = self.policy(s)
                # 分布は「当時のマスク」を厳密に再現できないため、学習時は近似的に全行動分布を使用
                # ※サンプル時の logprob を固定（old_logprobs）し、ratio の分子は現在の logprob
                dist = Categorical(logits=logits)
                new_logprob = dist.log_prob(a)

                ratio = (new_logprob - old_lp).exp()
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - CLIP_RANGE, 1.0 + CLIP_RANGE) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # 価値損失
                v = self.value(s)
                value_loss = nn.functional.mse_loss(v, ret)

                # エントロピー
                entropy = dist.entropy().mean()

                loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

                self.optim_policy.zero_grad()
                self.optim_value.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                nn.utils.clip_grad_norm_(self.value.parameters(), 1.0)
                self.optim_policy.step()
                self.optim_value.step()

    def _compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        T = rewards.size(0)
        advantages = torch.zeros(T, dtype=torch.float32)
        last_adv = 0.0
        for t in reversed(range(T)):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + GAMMA * values[t + 1] * nonterminal - values[t]
            last_adv = delta + GAMMA * LAMBDA * nonterminal * last_adv
            advantages[t] = last_adv
        returns = advantages + values[:-1]
        return advantages, returns

    def train(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        入力 df: columns=[Time, Price, Volume(累計)]
        戻り値 df_transaction: columns=[Time, Price, Action, Profit]
        """
        df = df.reset_index(drop=True)
        features, scaler = self._features_and_scaler(df)
        self.scaler = scaler  # 保存用に保持

        # ロールアウト収集
        states = []
        actions = []
        logprobs = []
        rewards = []
        dones = []
        values = []

        df_transaction = pd.DataFrame(columns=["Time", "Price", "Action", "Profit"])

        position = 0
        entry_price = 0.0

        for t in range(len(df)):
            price = float(df.loc[t, "Price"])
            state_np = features[t]  # (5,)
            states.append(state_np.copy())

            # ウォームアップまたは制約に応じて行動決定
            if t < WARMUP:
                action = ACTION_HOLD
                logprob = 0.0  # ダミー
                value_t = self.value(torch.tensor(state_np).float().unsqueeze(0)).item()
            else:
                state_t = torch.tensor(state_np).float().unsqueeze(0)
                logits = self.policy(state_t)
                mask = action_mask(position).unsqueeze(0)
                dist = masked_categorical(logits, mask)

                # ε-greedy：一定確率で許可行動からランダムに選択
                if np.random.rand() < EPSILON:
                    legal_indices = torch.where(mask[0])[0].cpu().numpy()
                    action = int(np.random.choice(legal_indices))
                    # 分布に基づく logprob（強制サンプルだが logprob は分布に沿って計算）
                    logprob = dist.log_prob(torch.tensor([action])).item()
                else:
                    action_t = dist.sample()
                    action = int(action_t.item())
                    logprob = dist.log_prob(torch.tensor([action])).item()

                value_t = self.value(state_t).item()

            # 取引ルールと報酬
            reward = 0.0
            realized = 0.0

            if action == ACTION_BUY and position == 0:
                position = 1
                entry_price = price + SLIPPAGE
            elif action == ACTION_SELL and position == 0:
                position = -1
                entry_price = price - SLIPPAGE
            elif action == ACTION_REPAY and position != 0:
                realized = realize_pnl(position, entry_price, price)
                reward += realized
                position = 0
                entry_price = 0.0
            else:
                # HOLD か、違反（マスクにより通常発生しない）
                pass

            # 保有中の含み益の一部を毎ティック報酬に反映（例: 10%）
            if position != 0:
                upnl = unrealized_pnl(position, entry_price, price)
                reward += 0.10 * upnl / max(1.0, UNIT)  # スケール過大化を緩和（調整可）

            # 終端判定は日中の1本ではFalse
            done = False

            actions.append(action)
            logprobs.append(logprob)
            rewards.append(float(reward))
            dones.append(done)
            values.append(float(value_t))

            # ログ
            df_transaction.loc[t] = [df.loc[t, "Time"], price, ACTION_MAP[action], float(realized if realized != 0 else reward)]

        # 終了時に建玉があれば強制返済
        if position != 0:
            price = float(df.loc[len(df) - 1, "Price"])
            force_realized = realize_pnl(position, entry_price, price)
            rewards[-1] += force_realized  # 最終報酬に加算
            df_transaction.loc[len(df_transaction)] = [df.loc[len(df)-1, "Time"], price, "REPAY", float(force_realized)]
            position = 0

        # ターミナルの value（bootstrap用に1つ余分）
        with torch.no_grad():
            last_v = 0.0  # 1日の最後でエピソード完結とみなす
        values.append(float(last_v))

        # テンソル化
        states_t = torch.tensor(np.array(states), dtype=torch.float32)
        actions_t = torch.tensor(np.array(actions), dtype=torch.long)
        old_logprobs_t = torch.tensor(np.array(logprobs), dtype=torch.float32)
        rewards_t = torch.tensor(np.array(rewards), dtype=torch.float32)
        dones_t = torch.tensor(np.array(dones), dtype=torch.float32)
        values_t = torch.tensor(np.array(values), dtype=torch.float32)

        # GAE / Returns
        advantages_t, returns_t = self._compute_gae(rewards_t, values_t, dones_t)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-9)

        data = {
            "states": states_t,
            "actions": actions_t,
            "logprobs": old_logprobs_t,
            "advantages": advantages_t,
            "returns": returns_t,
        }
        self._ppo_update(data)

        # モデル保存（スケーラーも一緒に）
        torch.save(
            {
                "state_dict": self.policy.state_dict(),
                "scaler_mean": self.scaler.mean,
                "scaler_std": self.scaler.std,
                "feature_dim": FEATURE_DIM,
            },
            self.policy_path,
        )
        torch.save(self.value.state_dict(), self.value_path)

        return df_transaction


# =========================================================
# TradingSimulator（推論専用：PC2）
# =========================================================
class RollingFeatures:
    """ ザラ場の逐次ティックから、必要特徴量をオンライン計算 """
    def __init__(self, window: int = WARMUP):
        self.window = window
        self.prices = deque(maxlen=window)
        self.gains = deque(maxlen=window)
        self.losses = deque(maxlen=window)
        self.last_price = None
        self.last_volume = None  # 累計出来高
        self.ready = False

    def update(self, price: float, cum_volume: float) -> Tuple[np.ndarray, bool]:
        # ΔVolume（累計からの差分）
        if self.last_volume is None:
            dvol = 0.0
        else:
            dvol = max(0.0, cum_volume - self.last_volume)
        self.last_volume = cum_volume

        # RSI 用
        if self.last_price is None:
            gain = 0.0
            loss = 0.0
        else:
            diff = price - self.last_price
            gain = max(0.0, diff)
            loss = max(0.0, -diff)
        self.last_price = price

        self.prices.append(price)
        self.gains.append(gain)
        self.losses.append(loss)

        if len(self.prices) < self.window:
            self.ready = False
            feats = np.zeros(FEATURE_DIM, dtype=np.float32)
            feats[0] = math.log1p(dvol)
            return feats, False

        self.ready = True
        arr = np.array(self.prices, dtype=np.float64)
        ma = float(arr.mean())
        std = float(arr.std(ddof=0))
        std = max(std, 1e-9)
        rsi = self._calc_rsi()
        z = (price - ma) / std

        feats = np.array([math.log1p(dvol), ma, std, rsi, z], dtype=np.float32)
        return feats, True

    def _calc_rsi(self) -> float:
        gains = np.array(self.gains, dtype=np.float64)
        losses = np.array(self.losses, dtype=np.float64)
        avg_gain = gains.mean()
        avg_loss = losses.mean()
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100.0 - 100.0 / (1.0 + rs)
        return float(rsi)


class TradingSimulator:
    def __init__(self, model_path: str = "policy.pth"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_path} が存在しません。PC1で学習後のモデルを配置してください。")
        ckpt = torch.load(model_path, map_location="cpu")
        self.scaler = FeatureScaler(mean=np.array(ckpt["scaler_mean"], dtype=np.float32),
                                    std=np.array(ckpt["scaler_std"], dtype=np.float32))
        feature_dim = ckpt.get("feature_dim", FEATURE_DIM)
        assert feature_dim == FEATURE_DIM, "特徴量次元が一致しません。"

        self.policy = PolicyNet(FEATURE_DIM)
        self.policy.load_state_dict(ckpt["state_dict"])
        self.policy.eval()

        self.fe = RollingFeatures(window=WARMUP)
        self.position = 0
        self.entry_price = 0.0

    @torch.no_grad()
    def add(self, time: float, price: float, volume: float) -> str:
        """
        time: 秒タイムスタンプ
        price: 現値
        volume: 累計出来高
        戻り値: "HOLD" | "BUY" | "SELL" | "REPAY"
        """
        feats, ready = self.fe.update(price, volume)
        if not ready:
            return "HOLD"

        x = self.scaler.transform(feats).astype(np.float32)
        x_t = torch.tensor(x).unsqueeze(0)
        logits = self.policy(x_t)
        mask = action_mask(self.position).unsqueeze(0)
        dist = masked_categorical(logits, mask)

        # ザラ場の推論は基本は貪欲: argmax（可：確率サンプリングに変更）
        probs = torch.softmax(dist.logits, dim=-1).squeeze(0).cpu().numpy()
        allowed = mask.squeeze(0).cpu().numpy()
        probs = probs * allowed
        if probs.sum() <= 0:
            action = ACTION_HOLD
        else:
            action = int(np.argmax(probs))

        # ルールに基づく状態更新
        if action == ACTION_BUY and self.position == 0:
            self.position = 1
            self.entry_price = price + SLIPPAGE
        elif action == ACTION_SELL and self.position == 0:
            self.position = -1
            self.entry_price = price - SLIPPAGE
        elif action == ACTION_REPAY and self.position != 0:
            # 返済するが、損益管理は自作アプリ側でやる想定。ここでは状態のみ更新。
            self.position = 0
            self.entry_price = 0.0

        return ACTION_MAP[action]


# =========================================================
# 動作例（ダミーデータで学習→保存→推論1ステップ）
# =========================================================
if __name__ == "__main__":
    # ダミーの1日データ生成（約 19,500 ティック）
    np.random.seed(42)
    n = 19_500
    times = np.arange(n, dtype=np.int64)
    prices = 1000 + np.cumsum(np.random.randn(n) * 0.5).astype(np.float64)
    volumes = np.cumsum(np.random.randint(1, 20, size=n)).astype(np.int64)
    df_day = pd.DataFrame({"Time": times, "Price": prices, "Volume": volumes})

    # 学習
    trainer = Trainer(policy_path="policy.pth", value_path="value.pth")
    df_tr = trainer.train(df_day)
    print(df_tr.head())
    print("総収益（Profit列合計）:", float(df_tr["Profit"].sum()))

    # 推論（policy.pth を使う）
    sim = TradingSimulator(model_path="policy.pth")
    # ザラ場 3 ティック分だけ例示
    print(sim.add(float(times[0]), float(prices[0]), float(volumes[0])))
    print(sim.add(float(times[1]), float(prices[1]), float(volumes[1])))
    print(sim.add(float(times[WARMUP]), float(prices[WARMUP]), float(volumes[WARMUP])))
