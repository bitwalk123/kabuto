import logging

import pandas as pd
from PySide6.QtCore import QObject, Signal, Slot

from modules.algo_trade import AlgoTrade
from modules.env import TradingEnv
from structs.app_enum import PositionType, ActionType


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

        self.list_obs_label = list()
        self.df_obs = None

        # 学習環境の取得
        self.env = TradingEnv(code, dict_param)

        # モデルのインスタンス
        self.model = AlgoTrade()

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
            obs, dict_technicals = self.env.getObservation(ts, price, volume)
            # 現在の行動マスクを取得
            masks = self.env.action_masks()
            # モデルによる行動予測
            action, _states = self.model.predict(obs, masks=masks)
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
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # 🧿 テクニカル指標を通知するシグナル
            self.sendTechnicals.emit(dict_technicals)
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

            # -----------------------------------------------------------------
            # obs をデータフレームへ追加
            for col, val in zip(self.list_obs_label, obs):
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
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 観測値を通知
        self.sendObs.emit(self.df_obs)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    @Slot()
    def getParams(self):
        dict_param = self.env.getParams()
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 テクニカル指標などのパラメータ取得
        self.sendParams.emit(dict_param)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    @Slot()
    def postProcs(self):
        dict_result = dict()
        dict_result["transaction"] = self.env.getTransaction()
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 売買履歴を通知
        self.sendResults.emit(dict_result)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    @Slot()
    def resetEnv(self):
        # 環境のリセット
        self.obs, _ = self.env.reset()
        self.done = False

        list_colname = ["Timestamp", "Price", "Volume"]
        self.list_obs_label = None
        self.list_obs_label = self.env.getObsList()
        self.model.updateObs(self.list_obs_label)

        list_colname.extend(self.list_obs_label)
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
