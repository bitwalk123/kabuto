import random

import pandas as pd
from talib import RSI

from modules.trading_env_20250906_copilot_1 import TradingEnv, ActionType

# ① ティックデータ読み込み
df = pd.read_excel("excel/tick_20250819.xlsx")

# ② 前処理関数（特徴量付与）
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df["MA"] = df["Price"].rolling(window=60).mean()
    df["STD"] = df["Price"].rolling(window=60).std()
    df["RSI"] = RSI(df["Price"], timeperiod=60)
    df["Zscore"] = (df["Price"] - df["MA"]) / df["STD"]
    df.ffill(inplace=True)
    return df

df = preprocess(df)

# ③ 環境構築
env = TradingEnv(df)

obs = env.reset()
done = False
while not done:
    action_enum = random.choice([ActionType.BUY, ActionType.SELL, ActionType.HOLD])
    action_idx = action_enum.value - 1  # Enum → int に変換
    obs, reward, terminated, truncated, info = env.step(action_idx)
    done = terminated or truncated
    print(f"Action: {action_enum}, Position: {env.position}, Reward: {reward}")
