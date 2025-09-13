import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from modules.trading_env_20250914 import TradingEnv

# ===== # 過去のティックデータ =====
file_excel = "excel/tick_20250828.xlsx"
df = pd.read_excel(file_excel)
print(df)

# ===== 環境初期化 =====
env = TradingEnv(df)

# ===== ランダムエージェント =====
obs, info = env.reset()
done = False
rewards = []
pnls = []
steps = 0

for step in range(100):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    print(step, [float(v) for v in obs])
