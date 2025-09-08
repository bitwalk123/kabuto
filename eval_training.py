import pandas as pd
import numpy as np

from modules.trading_env_20250908_1 import TradingEnv

# ダミーのティックデータを作成（本番では tick_xxxx.xlsx を読み込む）
n_ticks = 500
df = pd.DataFrame({
    "Time": np.arange(n_ticks),
    "Price": np.cumsum(np.random.randn(n_ticks)) + 100.0,
    "Volume": np.random.randint(1, 100, size=n_ticks)
})

# 環境を初期化
env = TradingEnv(df)

# エピソードを1回実行
obs, info = env.reset()
done = False

step_count = 0
total_reward = 0.0

while not done:
    action = env.action_space.sample()  # ランダムにアクションを選択
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    step_count += 1

    if step_count % 50 == 0:
        print(f"Step={step_count}, Reward={reward:.2f}, PnL={info['pnl_total']:.2f}")

print("Episode finished")
print("Total steps:", step_count)
print("Total reward:", total_reward)
print("Final PnL:", info["pnl_total"])
