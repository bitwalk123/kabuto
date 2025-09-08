import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from modules.ppo_agent_20250907 import PPOAgent, RolloutBuffer
from modules.trading_env_old import TradingEnv

# ===============================
# 1. データ読み込み
# ===============================
file_excel = "excel/tick_20250819.xlsx"
df = pd.read_excel(file_excel)

print(f"Loaded {len(df)} rows from {file_excel}")

# ===============================
# 2. 環境と PPO 初期化
# ===============================
env = TradingEnv(df)
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

agent = PPOAgent(obs_dim, n_actions, minibatch_size=128)

buffer = RolloutBuffer()
global_step = 0

# ハイパーパラメータ
#n_steps_per_update = 4096  # 1回の更新に必要なステップ数
n_steps_per_update = len(df)  # 1回の更新に必要なステップ数
max_epochs = 100  # 学習エポック数

reward_history = []
pnl_history = []

# ===============================
# 3. 学習ループ
# ===============================
for epoch in range(max_epochs):
    obs, _ = env.reset()
    done = False
    ep_rewards = 0.0
    last_info = {}

    while len(buffer.obs) < n_steps_per_update:
        action, logp, value = agent.select_action(obs)
        next_obs, reward, done, _, info = env.step(action)

        buffer.obs.append(obs)
        buffer.actions.append(action)
        buffer.logprobs.append(logp)
        buffer.rewards.append(reward)
        buffer.dones.append(float(done))
        buffer.values.append(value)

        obs = next_obs
        ep_rewards += reward
        global_step += 1
        last_info = info

        if done:
            obs, _ = env.reset()
            done = False

    # bootstrap value
    obs_t = torch.from_numpy(obs.astype(np.float32)).to(agent.device).unsqueeze(0)
    _, last_value = agent.net(obs_t)
    last_value = float(last_value.item())

    agent.update(buffer, last_value=last_value)

    # 記録
    reward_history.append(ep_rewards)
    pnl_history.append(last_info.get("pnl_total", 0.0))

    print(f"Epoch {epoch + 1}/{max_epochs} "
          f"steps={global_step} "
          f"episode_reward={ep_rewards:.2f} "
          f"pnl_total={last_info.get('pnl_total', 0.0):.2f}")
    #input("Please type eny key...")
    buffer.clear()

print("Training finished.")

# ===============================
# 4. 学習曲線の可視化
# ===============================
plt.figure(figsize=(12, 5))
plt.plot(reward_history, label="Episode Reward")
plt.plot(pnl_history, label="PnL Total")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Training Curve (PPO on TradingEnv with 19,000 ticks)")
plt.legend()
plt.grid(True)
plt.show()
