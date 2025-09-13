import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from modules.trading_env_20250913 import TradingEnv

# ===== # 過去のティックデータ =====
file_excel = "excel/tick_20250819.xlsx"
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

while not done:
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    rewards.append(reward)
    pnls.append(info["pnl_total"])
    steps += 1

print(
    f"Episode finished: steps={steps}, "
    f"total_reward={np.sum(rewards):.2f}, "
    f"final_pnl={pnls[-1]:.2f}"
)

# ===== 可視化 =====
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 報酬分布
axes[0].hist(rewards, bins=50, alpha=0.7)
# axes[0].hist(rewards, alpha=0.7)
axes[0].set_title("Reward distribution")
axes[0].set_xlabel("Reward")
axes[0].set_ylabel("Frequency")
# axes[0].set_yscale("log")
axes[0].grid()

# PnL 推移
axes[1].plot(pnls)
axes[1].set_title("PnL over steps")
axes[1].set_xlabel("Step")
axes[1].set_ylabel("Cumulative PnL")
axes[1].grid()

plt.tight_layout()
plt.show()
