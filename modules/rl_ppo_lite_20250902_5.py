import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


# -----------------------------
# ユーティリティ関数
# -----------------------------
def compute_features(df):
    df = df.copy()
    df["ΔVolume"] = df["Volume"].diff().fillna(0)
    df["logΔVolume"] = np.log1p(df["ΔVolume"])
    df["MA60"] = df["Price"].rolling(60).mean()
    df["STD60"] = df["Price"].rolling(60).std()
    df["RSI60"] = compute_rsi(df["Price"], n=60)
    df["Zscore60"] = (df["Price"] - df["MA60"]) / df["STD60"]
    df.fillna(0, inplace=True)
    return df[["logΔVolume", "MA60", "STD60", "RSI60", "Zscore60"]]


def compute_rsi(prices, n=60):
    delta = prices.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(n).mean()
    avg_loss = pd.Series(loss).rolling(n).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi


# -----------------------------
# PPOモデル
# -----------------------------
class PolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)


class ValueNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.fc(x).squeeze(-1)


# -----------------------------
# Trainer クラス
# -----------------------------
class Trainer:
    def __init__(self, model_path="policy.pth"):
        self.device = torch.device("cpu")
        self.policy = PolicyNet(input_dim=5, output_dim=4).to(self.device)
        self.value = ValueNet(input_dim=5).to(self.device)
        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=1e-4)
        self.optimizer_value = optim.Adam(self.value.parameters(), lr=1e-3)
        self.model_path = model_path

        if os.path.exists(model_path):
            self.policy.load_state_dict(torch.load(model_path))
            self.value.load_state_dict(torch.load(model_path.replace("policy", "value")))

    def train(self, df):
        df_features = compute_features(df)
        df_transaction = pd.DataFrame(columns=["Time", "Price", "Action", "Profit"])

        position = 0  # 0:無し, 1:ロング, -1:ショート
        entry_price = 0
        for t in range(len(df)):
            time, price = df.iloc[t]["Time"], df.iloc[t]["Price"]
            features = torch.tensor(df_features.iloc[t].values, dtype=torch.float32).unsqueeze(0)

            # ウォームアップ期間
            if t < 60:
                action = "HOLD"
            else:
                probs = self.policy(features).detach().numpy()[0]
                action_idx = np.random.choice(4, p=probs)
                action = ["HOLD", "BUY", "SELL", "REPAY"][action_idx]

                # 制約条件
                if position == 0 and action == "REPAY":
                    action = "HOLD"
                elif position != 0 and action in ["BUY", "SELL"]:
                    action = "HOLD"

            # 約定価格計算
            profit = 0
            if action == "BUY":
                entry_price = price + 1
                position = 1
            elif action == "SELL":
                entry_price = price - 1
                position = -1
            elif action == "REPAY":
                if position == 1:
                    profit = (price - 1 - entry_price) * 100
                elif position == -1:
                    profit = (entry_price - (price + 1)) * 100
                position = 0
                entry_price = 0

            # DataFrame へ記録
            df_transaction.loc[t] = [time, price, action, profit]

        # 最終行で建玉があれば強制返済（.loc で安全に代入）
        if position != 0:
            price = df.iloc[-1]["Price"]
            if position == 1:
                profit = (price - 1 - entry_price) * 100
            else:
                profit = (entry_price - (price + 1)) * 100
            df_transaction.loc[df_transaction.index[-1], "Action"] = "REPAY"
            df_transaction.loc[df_transaction.index[-1], "Profit"] = profit

        # 簡易的に PPO 更新処理（1エポック）
        returns = torch.tensor(df_transaction["Profit"].values, dtype=torch.float32)
        for t in range(len(df_transaction)):
            state = torch.tensor(df_features.iloc[t].values, dtype=torch.float32)
            action_taken = ["HOLD", "BUY", "SELL", "REPAY"].index(df_transaction.iloc[t]["Action"])
            prob = self.policy(state.unsqueeze(0))[0, action_taken]
            advantage = returns[t] - self.value(state)

            # Policy loss
            loss_policy = -torch.log(prob + 1e-8) * advantage.detach()
            self.optimizer_policy.zero_grad()
            loss_policy.backward()
            self.optimizer_policy.step()

            # Value loss
            loss_value = (self.value(state) - returns[t]) ** 2
            self.optimizer_value.zero_grad()
            loss_value.backward()
            self.optimizer_value.step()

        # モデル保存
        torch.save(self.policy.state_dict(), self.model_path)
        torch.save(self.value.state_dict(), self.model_path.replace("policy", "value"))

        return df_transaction


# -----------------------------
# TradingSimulator クラス
# -----------------------------
class TradingSimulator:
    def __init__(self, model_path="policy.pth"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_path} が存在しません。")
        self.device = torch.device("cpu")
        self.policy = PolicyNet(input_dim=5, output_dim=4).to(self.device)
        self.policy.load_state_dict(torch.load(model_path))
        self.df_buffer = pd.DataFrame(columns=["Time", "Price", "Volume"])
        self.position = 0
        self.entry_price = 0

    def add(self, time, price, volume):
        self.df_buffer.loc[len(self.df_buffer)] = [time, price, volume]
        df_features = compute_features(self.df_buffer)
        t = len(df_features) - 1
        if t < 60:
            return "HOLD"

        features = torch.tensor(df_features.iloc[t].values, dtype=torch.float32).unsqueeze(0)
        probs = self.policy(features).detach().numpy()[0]
        action_idx = np.random.choice(4, p=probs)
        action = ["HOLD", "BUY", "SELL", "REPAY"][action_idx]

        # 制約条件
        if self.position == 0 and action == "REPAY":
            action = "HOLD"
        elif self.position != 0 and action in ["BUY", "SELL"]:
            action = "HOLD"

        # 約定価格計算（保持）
        if action == "BUY":
            self.entry_price = price + 1
            self.position = 1
        elif action == "SELL":
            self.entry_price = price - 1
            self.position = -1
        elif action == "REPAY":
            self.position = 0
            self.entry_price = 0

        return action


# -----------------------------
# 参考：学習用
# -----------------------------
if __name__ == "__main__":
    list_excel = [
        "tick_20250819.xlsx",
        "tick_20250820.xlsx",
    ]
    df_lc = pd.DataFrame(columns=["Epoch", "Data", "Profit"])
    epoch = 0
    for file_excel in list_excel:
        df = pd.read_excel(file_excel)  # get_excel_sheet の代替
        trainer = Trainer()
        df_transaction = trainer.train(df)
        profit = df_transaction["Profit"].sum()
        print(f"Epoch: {epoch}, {file_excel}, 総収益: {profit}")
        df_transaction.to_csv(f"trade_results_{epoch:03}.csv")
        df_lc.loc[epoch] = [epoch, file_excel, profit]
        epoch += 1
    df_lc.to_csv("learning_curve.csv", index=False)
