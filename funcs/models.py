import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from structs.res import AppRes


def get_ppo_model_new(env: DummyVecEnv) -> PPO:
    # PPO モデルの生成
    # LSTM を含む方策ネットワーク MlpLstmPolicy を指定
    #return RecurrentPPO("MlpLstmPolicy", env, verbose=1)
    return PPO("MlpPolicy", env, verbose=1)


def get_ppo_model_path(res: AppRes, code: str) -> str:
    model_path = os.path.join(res.dir_model, f"ppo_{code}.zip")
    return model_path


def get_trained_ppo_model_path(res: AppRes, code: str) -> str:
    model_path = os.path.join(res.dir_model, "trained", f"ppo_{code}.zip")
    return model_path
