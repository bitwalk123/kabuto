# train_with_excel.py
import pandas as pd
import numpy as np

from modules.ppo_agent_20250905_4 import PPOAgent
from modules.trading_env_20250925_4 import TradingEnv

if __name__ == "__main__":
    # === Excel 読み込み ===
    # tick_20250819.xlsx のシートは 1枚目を想定
    df = pd.read_excel("excel/tick_20250819.xlsx")

    # 必要な列があるか確認（Time, Price, Volume）
    required_cols = {"Time", "Price", "Volume"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Excel ファイルに必要な列 {required_cols} が見つかりません。")

    # NaN や dtype の調整
    df = df.dropna(subset=["Price", "Volume"]).reset_index(drop=True)
    df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

    # === 環境初期化 ===
    env = TradingEnv(df)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    agent = PPOAgent(obs_dim, act_dim)

    # === 学習ループ（スモークテスト） ===
    #max_episodes = 5     # 短くテスト
    #horizon = 256        # ステップ数も短め
    horizon = 1024  # 256 → 1024
    max_episodes = 10  # 少し長めに

    for ep in range(max_episodes):
        obs, _ = env.reset()
        obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
        ep_reward = 0

        for t in range(horizon):
            act, logp, val = agent.select_action(obs)
            next_obs, rew, done, _, _ = env.step(act)

            obs_buf.append(obs)
            act_buf.append(act)
            logp_buf.append(logp)
            val_buf.append(val)
            rew_buf.append(rew)
            done_buf.append(done)

            obs = next_obs
            ep_reward += rew
            if done:
                break

            if t % 200 == 0:  # 200ステップごとにログ
                print(f"t={t}, act={act}, reward={rew}, pos={env.position}, pnl={env.get_realized_pnl()}")

        # bootstrap value
        import torch
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
            _, last_val = agent.net(obs_t)
            last_val = last_val.item()

        # returns と advantages を計算
        ret_buf = agent.compute_returns(rew_buf, done_buf, last_val)
        adv_buf = np.array(ret_buf) - np.array(val_buf)

        # 学習更新
        agent.update(obs_buf, act_buf, logp_buf, ret_buf, adv_buf)

        print(f"[Episode {ep+1}] total reward={ep_reward:.2f}, realized pnl={env.get_realized_pnl():.2f}")
