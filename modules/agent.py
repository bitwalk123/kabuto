import logging
from collections import defaultdict

import pandas as pd
from PySide6.QtCore import QObject, Signal, Slot

from funcs.tide import conv_datetime_from_timestamp
from modules.algo_trade import AlgoTrade
from modules.env import TradingEnv
from modules.posman import PositionManager
from structs.app_enum import ActionType, PositionType


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


class WorkerAgentRT(QObject):
    """
    強化学習を利用せずに、アルゴリズムのみのエージェント
    （リアルタイム用）
    """
    completedResetEnv = Signal()
    completedTrading = Signal()
    notifyAction = Signal(int, PositionType)  # 売買アクションを通知
    sendTechnicals = Signal(dict)

    """
    readyNext = Signal()
    sendObs = Signal(pd.DataFrame)
    sendParams = Signal(dict)
    sendResults = Signal(dict)
    """

    def __init__(self, autopilot: bool, code: str, dict_param: dict):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.autopilot = autopilot

        self.obs = None
        self.done = False

        self.list_obs = list()

        # 学習環境の取得
        self.env = TradingEnv(code, dict_param)

        # モデルのインスタンス
        self.model = AlgoTrade(self.list_obs)

    @Slot(float, float, float)
    def addData(self, ts: float, price: float, volume: float):
        if not self.done:
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
                pass

    @Slot()
    def forceRepay(self):
        self.env.forceRepay()

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


class CronAgent:
    """
    cron で実行できる GUI を利用しないエージェント
    """

    def __init__(self, code: str, dict_ts: dict):
        self.logger = logging.getLogger(__name__)
        self.code = code
        self.ts_end = dict_ts["end"]

        # モデルのインスタンス
        self.list_obs = list()
        self.model = AlgoTrade(self.list_obs)

        # ポジション・マネージャ
        self.posman = PositionManager()
        self.posman.initPosition([code])

        # 環境クラス
        self.env: TradingEnv | None = None

        # 取引内容
        self.dict_list_tech = defaultdict(list)

    def run(self, dict_param: dict, df: pd.DataFrame):
        # 環境の定義
        self.env = TradingEnv(self.code, dict_param)

        # 環境のリセット
        self.resetEnv()

        # データフレームの行数分のループ
        ts = 0
        price = 0
        for row in df.itertuples():
            ts = row.Time
            if self.ts_end < ts:
                break
            price = row.Price
            volume = row.Volume
            if self.addData(ts, price, volume):
                break

        # ポジション解消
        self.forceClosePosition(ts, price)

    def addData(self, ts: float, price: float, volume: float) -> bool:
        # ティックデータから観測値を取得
        obs, dict_technicals = self.env.getObservation(ts, price, volume)

        # 現在の行動マスクを取得
        masks = self.env.action_masks()

        # モデルによる行動予測
        action, _states = self.model.predict(obs, masks=masks)

        # self.autopilot フラグが立っていればアクションとポジションを通知
        # if self.autopilot:
        position: PositionType = self.env.getCurrentPosition()
        if ActionType(action) != ActionType.HOLD:
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # 🧿 売買アクションを通知するシグナル（HOLD の時は通知しない）
            # self.notifyAction.emit(action, position)
            self.on_action(ts, price, action, position)
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # -----------------------------------------------------------------
        # プロット用テクニカル指標
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 テクニカル指標を通知するシグナル
        # self.sendTechnicals.emit(dict_technicals)
        for key, value in dict_technicals.items():
            self.dict_list_tech[key].append(value)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # -----------------------------------------------------------------
        # アクションによる環境の状態更新
        # 【注意】 リアルタイム用環境では step メソッドで観測値は返されない
        # -----------------------------------------------------------------
        reward, terminated, truncated, info = self.env.step(action)
        if terminated:
            print("terminated フラグが立ちました。")
            return True
        elif truncated:
            print("truncated フラグが立ちました。")
            return True
        else:
            return False

    def getTechnicals(self) -> pd.DataFrame:
        df = pd.DataFrame(self.dict_list_tech)
        # インデックスを日付形式に変換
        df.index = [pd.to_datetime(conv_datetime_from_timestamp(ts)) for ts in df["ts"]]
        return df

    def getTransaction(self) -> pd.DataFrame:
        return self.posman.getTransactionResult()

    def on_action(self, ts: float, price: float, action: int, position: PositionType):
        action_enum = ActionType(action)
        if action_enum == ActionType.BUY:
            if position == PositionType.NONE:
                # 建玉がなければ買建
                self.posman.openPosition(self.code, ts, price, action_enum)
            elif position == PositionType.SHORT:
                # 売建（ショート）であれば（買って）返済
                self.posman.closePosition(self.code, ts, price)
            else:
                self.logger.error(f"{__name__}: trade rule violation!")
        elif action_enum == ActionType.SELL:
            if position == PositionType.NONE:
                # 建玉がなければ売建
                self.posman.openPosition(self.code, ts, price, action_enum)
            elif position == PositionType.LONG:
                # 買建（ロング）であれば（売って）返済
                self.posman.closePosition(self.code, ts, price)
            else:
                self.logger.error(f"{__name__}: trade rule violation!")
        elif action_enum == ActionType.HOLD:
            pass
        else:
            self.logger.error(f"{__name__}: unknown action type {action_enum}!")

    def forceClosePosition(self, ts: float, price: float):
        position: PositionType = self.env.getCurrentPosition()
        if position != PositionType.NONE:
            # ポジションがあれば返済
            self.posman.closePosition(self.code, ts, price)

    def resetEnv(self):
        # 環境のリセット
        obs, _ = self.env.reset()

        list_colname = ["Timestamp", "Price", "Volume"]
        self.list_obs.clear()
        self.list_obs.extend(self.env.getObsList())

        list_colname.extend(self.list_obs)
        dict_colname = dict()
        for colname in list_colname:
            dict_colname[colname] = []
