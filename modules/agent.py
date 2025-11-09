import os

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from modules.agent_auxiliary import ActionMaskWrapper
from modules.env import TrainingEnv


class PPOAgentSB3:
    def __init__(self):
        super().__init__()
        # ラップしないオリジナルの環境保持用
        self.env_raw = None
        # 結果保持用辞書
        self.results = dict()
        # 設定値
        self.total_timesteps = 100_000

    def get_env_with_df(self, df: pd.DataFrame) -> Monitor:
        # 環境のインスタンスを生成
        self.env_raw = env_raw = TrainingEnv(df)
        # ActionMaskWrapper ラッパーの適用
        env = ActionMaskWrapper(env_raw)
        # SB3の環境チェック（オプション）
        check_env(env, warn=True)
        # Monitor ラッパーの適用
        env_monitor = Monitor(env)

        return env_monitor

    def train(self, df: pd.DataFrame, path_model: str, log_dir: str, new_model: bool = False):
        custom_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])  # 出力形式を指定

        # 学習環境の取得
        env = self.get_env_with_df(df)
        # 学習済モデルを読み込む
        if not new_model and os.path.exists(path_model):
            print(f"モデル {path_model} を読み込みます。")
            try:
                model = PPO.load(path_model, env, verbose=1)
            except ValueError:
                print("読み込み時、例外 ValueError が発生したので新規にモデルを作成します。")
                model = PPO("MlpPolicy", env, verbose=1)
        else:
            print(f"新規にモデルを作成します。")
            model = PPO("MlpPolicy", env, verbose=1)

        # ロガーを差し替え
        model.set_logger(custom_logger)

        # モデルの学習
        model.learn(total_timesteps=self.total_timesteps)

        # モデルの保存
        print(f"モデルを {path_model} に保存します。")
        model.save(path_model)

        # 学習環境の解放
        env.close()

    def infer(self, df: pd.DataFrame, path_model: str) -> bool:
        # 学習環境の取得
        env = self.get_env_with_df(df)

        # 学習済モデルを読み込む
        if os.path.exists(path_model):
            print(f"モデル {path_model} を読み込みます。")
        else:
            print(f"モデルを {path_model} がありませんでした。")
            return False
        try:
            model = PPO.load(path_model, env, verbose=1)
        except ValueError as e:
            print(e)
            return False

        self.results["obs"] = list()
        self.results["reward"] = list()
        obs, _ = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            # 観測値トレンド成用
            self.results["obs"].append(obs)
            # 報酬分布作成用
            self.results["reward"].append(reward)

        # 取引内容
        self.results["transaction"] = self.env_raw.getTransaction()

        # 学習環境の解放
        env.close()

        return True
