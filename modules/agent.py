import logging
import os

import pandas as pd
from PySide6.QtCore import QObject, Signal, Slot
from sb3_contrib import MaskablePPO
from stable_baselines3.common.logger import configure

from modules.env import TrainingEnv, TradingEnv
from structs.app_enum import ActionType, PositionType


class MaskablePPOAgent:
    def __init__(self):
        super().__init__()
        # 結果保持用辞書
        self.results = dict()
        # 設定値
        self.total_timesteps = 100_000

    def train(self, df: pd.DataFrame, path_model: str, log_dir: str, new_model: bool = False):
        # 出力形式を指定
        custom_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

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

    def infer(self, df: pd.DataFrame, path_model: str, flag_all: bool = False) -> bool:
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


class WorkerAgent(QObject):
    # 売買アクションを通知
    notifyAction = Signal(int, PositionType)
    finished = Signal()

    def __init__(self, path_model: str, autopilot: bool):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.done = False
        self.autopilot = autopilot
        self._running = True
        self._stop_flag = False
        self.logger.info(f"{__name__}: model, {path_model} is used.")

        # 学習環境の取得
        self.env = env = TradingEnv()
        self.obs, _ = env.reset()
        # 学習済モデルの読み込み
        self.model = MaskablePPO.load(path_model, env)

    @Slot(float, float, float)
    def addData(self, ts: float, price: float, volume: float) -> bool:
        if not self.done:
            # マスク情報を取得
            masks = self.env.action_masks()
            # モデルによる行動予測
            action, _states = self.model.predict(self.obs, action_masks=masks)

            # self.autopilot フラグが立っていれば通知
            if self.autopilot:
                position: PositionType = self.env.reward_man.position
                if ActionType(action) != ActionType.HOLD:
                    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    # 売買アクションを通知するシグナル（HOLD の時は通知しない）
                    self.notifyAction.emit(action, position)
                    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++

            # マスク更新と報酬計算
            self.env.setData(ts, price, volume)
            self.obs, reward, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                self.done = True

        return self.done

    def getTransaction(self) -> pd.DataFrame:
        return self.env.getTransaction()

    @Slot(bool)
    def setAutoPilotStatus(self, state: bool):
        self.autopilot = state
        self.logger.info(f"{__name__}: autopilot is set to {state}.")

    @Slot()
    def stop(self):
        """終了処理"""
        self._stop_flag = True
        self.finished.emit()
