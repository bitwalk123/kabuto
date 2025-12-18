import logging
import os

import pandas as pd
from PySide6.QtCore import QObject, Signal, Slot
from sb3_contrib import MaskablePPO
from stable_baselines3.common.logger import configure

from modules.algo_trade import AlgoTrade
from modules.env import TrainingEnv, TradingEnv
from structs.app_enum import ActionType, PositionType


class MaskablePPOAgent:
    def __init__(self):
        super().__init__()
        self.env = None
        # 結果保持用辞書
        self.results = dict()
        # 設定値
        self.total_timesteps = 100_000

    def train(self, df: pd.DataFrame, path_model: str, log_dir: str, new_model: bool = False):
        # 出力形式を指定
        custom_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

        # 学習環境の取得
        self.env = env = TrainingEnv(df)
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
        self.env = env = TrainingEnv(df)

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


class WorkerAgentSB3(QObject):
    completedResetEnv = Signal()
    completedTrading = Signal()
    notifyAction = Signal(int, PositionType)  # 売買アクションを通知
    readyNext = Signal()
    sendResults = Signal(dict)

    def __init__(self, path_model: str, autopilot: bool):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.obs = None
        self.done = False
        self.autopilot = autopilot

        # 学習環境の取得
        self.env = env = TradingEnv()
        # 学習済モデルの読み込み
        self.logger.info(f"{__name__}: model, {path_model} is used.")
        self.model = MaskablePPO.load(path_model, env)

    @Slot(float, float, float)
    def addData(self, ts: float, price: float, volume: float):
        if self.done:
            # 取引終了（念の為）
            self.completedTrading.emit()
        else:
            # ティックデータから観測値を取得
            obs = self.env.getObservation(ts, price, volume)
            # 現在の行動マスクを取得
            masks = self.env.action_masks()
            # モデルによる行動予測
            action, _states = self.model.predict(obs, action_masks=masks)

            # self.autopilot フラグが立っていればアクションとポジションを通知
            if self.autopilot:
                position: PositionType = self.env.reward_man.position
                if ActionType(action) != ActionType.HOLD:
                    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    # 売買アクションを通知するシグナル（HOLD の時は通知しない）
                    self.notifyAction.emit(action, position)
                    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++

            # -----------------------------------------------------------------
            # アクションによる環境の状態更新
            # 【注意】 リアルタイム用環境では step メソッドで観測値は返されない
            # -----------------------------------------------------------------
            reward, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                self.done = True
                # 取引終了
                self.completedTrading.emit()
            else:
                # 次のアクション受け入れ準備完了
                self.readyNext.emit()

    def postProcs(self):
        dict_result = dict()
        dict_result["transaction"] = self.env.getTransaction()
        self.sendResults.emit(dict_result)

    @Slot()
    def resetEnv(self):
        # 環境のリセット
        self.obs, _ = self.env.reset()
        self.done = False
        self.completedResetEnv.emit()

    @Slot(bool)
    def setAutoPilotStatus(self, state: bool):
        self.autopilot = state
        self.logger.info(f"{__name__}: autopilot is set to {state}.")


class WorkerAgent(QObject):
    """
    強化学習を利用せずに、アルゴリズムのみのエージェント
    """
    completedResetEnv = Signal()
    completedTrading = Signal()
    notifyAction = Signal(int, PositionType)  # 売買アクションを通知
    readyNext = Signal()
    sendObs = Signal(pd.DataFrame)
    sendParams = Signal(dict)
    sendResults = Signal(dict)
    sendTechnicals = Signal(dict)

    def __init__(self, autopilot: bool, code: str, dict_param: dict):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.autopilot = autopilot

        self.obs = None
        self.done = False

        self.list_obs = list()
        self.df_obs = None

        # 学習環境の取得
        self.env = TradingEnv(code, dict_param)

        # モデルのインスタンス
        self.model = AlgoTrade(self.list_obs)

    @Slot(float, float, float)
    def addData(self, ts: float, price: float, volume: float):
        if self.done:
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # 🧿 取引終了（念の為）
            self.completedTrading.emit()
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        else:
            # ティックデータをデータフレームへ追加
            row = len(self.df_obs)
            self.df_obs.at[row, "Timestamp"] = ts
            self.df_obs.at[row, "Price"] = price
            self.df_obs.at[row, "Volume"] = volume
            # ティックデータから観測値を取得
            obs = self.env.getObservation(ts, price, volume)
            # 現在の行動マスクを取得
            masks = self.env.action_masks()
            # モデルによる行動予測
            action, _states = self.model.predict(obs, action_masks=masks)
            # self.autopilot フラグが立っていればアクションとポジションを通知
            if self.autopilot:
                position: PositionType = self.env.getCurrentPosition()
                if ActionType(action) != ActionType.HOLD:
                    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                    # 🧿 売買アクションを通知するシグナル（HOLD の時は通知しない）
                    self.notifyAction.emit(action, position)
                    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++

            # -----------------------------------------------------------------
            # プロット用テクニカル指標
            dict_tech = {"ts": ts, "ma_1": float(obs[0]), "ma_2": float(obs[1])}
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # 🧿 テクニカル指標を通知するシグナル
            self.sendTechnicals.emit(dict_tech)
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

            # -----------------------------------------------------------------
            # obs をデータフレームへ追加
            for col, val in zip(self.list_obs, obs):
                self.df_obs.at[row, col] = val
            # -----------------------------------------------------------------
            # アクションによる環境の状態更新
            # 【注意】 リアルタイム用環境では step メソッドで観測値は返されない
            # -----------------------------------------------------------------
            reward, terminated, truncated, info = self.env.step(action)
            if terminated:
                print("terminated フラグが立ちました。")
                self.done = True
                # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # 🧿 取引終了
                self.completedTrading.emit()
                # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            elif truncated:
                print("truncated フラグが立ちました。")
                self.done = True
                # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # 🧿 取引終了
                self.completedTrading.emit()
                # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            else:
                # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # 🧿 次のアクション受け入れ準備完了
                self.readyNext.emit()
                # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    @Slot()
    def forceRepay(self):
        self.env.forceRepay()

    @Slot()
    def getObs(self):
        self.sendObs.emit(self.df_obs)

    @Slot()
    def getParams(self):
        dict_param = self.env.getParams()
        # テクニカル指標などのパラメータ取得
        self.sendParams.emit(dict_param)

    @Slot()
    def postProcs(self):
        dict_result = dict()
        dict_result["transaction"] = self.env.getTransaction()
        self.sendResults.emit(dict_result)

    @Slot()
    def resetEnv(self):
        # 環境のリセット
        self.obs, _ = self.env.reset()
        self.done = False

        list_colname = ["Timestamp", "Price", "Volume"]
        self.list_obs.clear()
        self.list_obs.extend(self.env.getObsList())
        list_colname.extend(self.list_obs)
        dict_colname = dict()
        for colname in list_colname:
            dict_colname[colname] = []
        self.df_obs = pd.DataFrame(dict_colname)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 環境のリセット環境を通知
        self.completedResetEnv.emit()
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    @Slot(bool)
    def setAutoPilotStatus(self, state: bool):
        self.autopilot = state
        self.logger.info(f"{__name__}: autopilot is set to {state}.")
