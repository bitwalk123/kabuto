import logging
import os

import pandas as pd
from PySide6.QtCore import QObject, Signal, Slot
from sb3_contrib import MaskablePPO
from stable_baselines3.common.logger import configure

from modules.env import TrainingEnv, TradingEnv, PositionType


class PPOAgentSB3:
    def __init__(self):
        super().__init__()
        # 結果保持用辞書
        self.results = dict()
        # 設定値
        self.total_timesteps = 100_000

    def train(self, df: pd.DataFrame, path_model: str, log_dir: str, new_model: bool = False):
        custom_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])  # 出力形式を指定

        # 学習環境の取得
        env = TrainingEnv(df)
        # 学習済モデルを読み込む
        if not new_model and os.path.exists(path_model):
            print(f"モデル {path_model} を読み込みます。")
            try:
                model = MaskablePPO.load(path_model, env, verbose=1)
            except ValueError:
                print("読み込み時、例外 ValueError が発生したので新規にモデルを作成します。")
                model = MaskablePPO("MlpPolicy", env, verbose=1)
        else:
            print(f"新規にモデルを作成します。")
            model = MaskablePPO("MlpPolicy", env, verbose=1)

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
        env = TrainingEnv(df)

        # 学習済モデルを読み込む
        if os.path.exists(path_model):
            print(f"モデル {path_model} を読み込みます。")
        else:
            print(f"モデルを {path_model} がありませんでした。")
            return False
        try:
            model = MaskablePPO.load(path_model, env, verbose=1)
        except ValueError as e:
            print(e)
            return False

        self.results["obs"] = list()
        self.results["reward"] = list()
        obs, _ = env.reset()
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action_masks = env.action_masks()
            action, _states = model.predict(obs, action_masks=action_masks)
            obs, reward, terminated, truncated, info = env.step(action)
            # 観測値トレンド成用
            self.results["obs"].append(obs)
            # 報酬分布作成用
            self.results["reward"].append(reward)

        # 取引内容
        self.results["transaction"] = env.getTransaction()

        # 学習環境の解放
        env.close()

        return True


class AgentWorker(QObject):
    # 売買アクションを通知
    notifyAction = Signal(int, PositionType)
    finished = Signal()

    def __init__(self, path_model: str, autopilot: bool):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.autopilot = autopilot
        self._running = True
        self._stop_flag = False

        # 学習環境の取得
        self.env = env = TradingEnv()
        env.reset()
        # 学習済モデルの読み込み
        self.model = MaskablePPO.load(path_model, env)

    @Slot(float, float, float)
    def addData(self, ts, price, volume):
        obs = self.env.receive_tick(ts, price, volume)  # 状態更新のみ
        action, _ = self.model.predict(obs)
        action_masks = self.env.action_masks()  # マスク情報を取得
        action, _states = self.model.predict(obs, action_masks=action_masks)

        position: PositionType = self.env.trans_man.position
        if self.autopilot:
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # 🧿 売買アクションを通知するシグナル
            self.notifyAction.emit(action, position)
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        obs, reward, _, _, info = self.env.step(action)  # マスク更新と報酬計算

    @Slot(bool)
    def setAutoPilotStatus(self, state: bool):
        self.autopilot = state
        self.logger.info(f"{__name__}: autopilot is set to {state}.")

    @Slot()
    def stop(self):
        """終了処理"""
        self._stop_flag = True
        self.finished.emit()
