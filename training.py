import os
import random

import numpy as np
import pandas as pd
import torch
from torch import optim as optim

from modules.ppo_agent import (
    ActorCritic,
    RunningMeanStd,
    compute_gae,
    ppo_update
)
from modules.trading_env import TradingEnv


class PPOAgent:
    def __init__(
            self,
            env: TradingEnv,
            lr=3e-4,
            gamma=0.99,
            lam=0.95,
            clip_eps=0.2,
            batch_size=64,
            hidden_sizes=(256, 256)
    ):
        self.env = env

        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.batch_size = batch_size

        # ネットワーク
        obs_dim = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n
        self.model = ActorCritic(obs_dim, n_actions, hidden_sizes)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # バッファ
        # self.reset_buffer()


def train_on_file(env: TradingEnv, dir_model: str, dir_output: str, n_epochs: int = 100, seed: int = 0):
    """
    model training
    :param env:
    :param dir_model:
    :param dir_output:
    :param n_epochs:
    :param seed:
    :return:
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ActorCritic(obs_dim, n_actions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
    # optimizer = optim.Adam(model.parameters(), lr=1e-4, eps=1e-5)

    # hyperparams (tuned to be stable for single-episode rollouts)
    gamma = 0.99
    lam = 0.95
    clip = 0.2
    ppo_epochs = 8
    minibatch_size = 128
    ent_coef = 0.01
    vf_coef = 0.5

    obs_rms = RunningMeanStd(shape=(obs_dim,))

    # storage for logs
    history = {
        'epoch': [],
        'episode_reward': [],
        'pnl_total': [],
        'approx_kl': [],
        'clipfrac': [],
        'transactions': [],
    }

    for epoch in range(1, n_epochs + 1):
        # run 1 full episode (one file -> one episode)
        obs_list, actions_list, rewards_list, dones_list, values_list, logp_list = [], [], [], [], [], []

        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        step = 0
        while not done:
            obs_norm = (obs - obs_rms.mean) / (np.sqrt(obs_rms.var) + 1e-8)
            obs_tensor = torch.tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, value = model(obs_tensor)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample().cpu().numpy()[0]
                logp = dist.log_prob(torch.tensor(action)).cpu().numpy()
                value = value.cpu().numpy()[0]

            next_obs, reward, done, truncated, info = env.step(int(action))

            obs_list.append(obs)
            actions_list.append(action)
            rewards_list.append(reward)
            dones_list.append(done)
            values_list.append(value)
            logp_list.append(logp)

            total_reward += reward
            obs = next_obs
            step += 1

        # update running obs stats
        obs_rms.update(np.array(obs_list))

        # compute last value for bootstrap
        with torch.no_grad():
            last_obs_norm = (obs - obs_rms.mean) / (np.sqrt(obs_rms.var) + 1e-8)
            last_obs_t = torch.tensor(
                last_obs_norm,
                dtype=torch.float32,
                device=device
            ).unsqueeze(0)
            _, last_value = model(last_obs_t)
            last_value = last_value.cpu().numpy()[0]

        # compute GAE advantages and returns
        values_arr = np.asarray(values_list, dtype=np.float32)
        advantages, returns = compute_gae(
            np.asarray(rewards_list, dtype=np.float32),
            values_arr,
            np.asarray(dones_list, dtype=np.float32),
            last_value,
            gamma=gamma,
            lam=lam
        )

        # prepare logprobs_old as float array
        logp_arr = np.asarray(logp_list, dtype=np.float32)

        # normalize observations at training time
        obs_arr = np.asarray(
            [(o - obs_rms.mean) / (np.sqrt(obs_rms.var) + 1e-8) for o in obs_list],
            dtype=np.float32
        )

        # PPO update
        approx_kl, clipfrac = ppo_update(
            model,
            optimizer,
            obs_arr,
            actions_list,
            logp_arr,
            returns,
            advantages,
            clip_coef=clip,
            vf_coef=vf_coef,
            ent_coef=ent_coef,
            max_grad_norm=0.5,
            epochs=ppo_epochs,
            minibatch_size=minibatch_size
        )

        history['epoch'].append(epoch)
        history['episode_reward'].append(total_reward)
        history['pnl_total'].append(env.transman.pnl_total)

        """
        Kullback–Leibler divergence
        意味
        - 新しい方策 π_θ と古い方策 π_θ_old の「確率分布の違い」を測る指標。
        直感
        - 値が大きい → 新しい方策が古い方策から大きく乖離してしまった。
        - 値が小さい → 方策更新は穏やかで安定している。
        目安
        - 0.01 前後なら「安全」。
        - 0.05 以上に跳ね上がると「更新幅が大きすぎる（崩壊の危険あり）」とされ、学習率や clip を調整する必要あり。
        """
        history['approx_kl'].append(approx_kl)

        """
        Clipping Fraction
        意味
        - PPO の クリッピング項 が「どれくらいの割合のサンプルで効いたか」を表す。
        直感
        - 値が高い（例: 0.4 以上）
          → 多くのサンプルがクリップされ、更新が制限されている → 学習率や batch が大きすぎる可能性。
        - 値が低い（例: 0.05 未満）
          → ほとんどのサンプルが clip 範囲内で無難に更新されている → 学習が停滞しているかもしれない。
        目安
        - 0.1〜0.3 あたりが「健康なゾーン」と言われる。
        """
        history['clipfrac'].append(clipfrac)

        # 取引明細
        df_transaction = pd.DataFrame(env.transman.dict_transaction)
        file_transaction = os.path.join(dir_output, f"transaction_{epoch:03d}.csv")
        df_transaction.to_csv(file_transaction, index=False)

        n_transaction = len(df_transaction)
        history["transactions"].append(n_transaction)

        print(
            f"Epoch {epoch:03d} | "
            f"Steps {step:05d} | "
            f"Reward {total_reward:+12.1f} | "
            f"PnL {env.transman.pnl_total:+6.1f} | "
            f"KL {approx_kl:.5f} | "
            f"ClipFrac {clipfrac:.3f} | "
            f"Transactions {n_transaction: 5d}"
        )

        # save model every 10 epochs
        if epoch % 10 == 0:
            fname = os.path.join(dir_model, f"ppo_trading_epoch{epoch}.pt")
            torch.save(model.state_dict(), fname)

    # final save
    torch.save(model.state_dict(), os.path.join(dir_model, "ppo_trading_final.pt"))

    # write history to CSV
    hist_df = pd.DataFrame(history)
    file_history = os.path.join(dir_output, "training_history.csv")
    hist_df.to_csv(file_history, index=False)
    print(f"\nTraining finished. History saved to {file_history}")


if __name__ == '__main__':
    # Path to your tick data file
    dir_excel = "excel"
    path_excel = os.path.join(dir_excel, "tick_20250819.xlsx")
    if not os.path.exists(path_excel):
        raise FileNotFoundError(f"{path_excel} not found in working directory")

    df = pd.read_excel(path_excel)
    env = TradingEnv(df)
    dir_model = "models"
    dir_result = "output"
    # training
    train_on_file(env, dir_model, dir_result, n_epochs=3, seed=12345)
