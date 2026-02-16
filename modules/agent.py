import logging
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from PySide6.QtCore import (
    QObject,
    Signal,
    Slot,
)

from funcs.tide import conv_datetime_from_timestamp
from modules.algo_trade import AlgoTrade
from modules.env import TradingEnv
from modules.posman import PositionManager
from structs.app_enum import ActionType, PositionType


class WorkerAgentRT(QObject):
    """
    強化学習を利用せずに、アルゴリズムのみのエージェント
    （リアルタイム用）
    """
    completedResetEnv = Signal()
    completedTrading = Signal()
    notifyAction = Signal(int, PositionType)  # 売買アクションを通知
    sendTechnicals = Signal(dict)

    def __init__(self, code: str, dict_param: dict[str, Any]) -> None:
        super().__init__()
        self.logger = logging.getLogger(__name__)

        self.obs = None
        self.done: bool = False
        self.df_obs = None
        self._is_stopping: bool = False  # 終了フラグを追加

        # 学習環境の取得
        self.env = TradingEnv(code, dict_param)

        # モデルに渡す観測値のリスト
        self.list_obs_label: list[str] = []
        # モデルのインスタンス
        self.model = AlgoTrade()

        # 取引内容（＋テクニカル指標）
        self.dict_list_tech: dict[str, list] = defaultdict(list)

    @Slot(float, float, float)
    def addData(self, ts: float, price: float, volume: float) -> None:
        # 終了処理中はデータを処理しない
        if self._is_stopping:
            return

        if not self.done:
            # ティックデータから観測値を取得
            obs, dict_technicals = self.env.getObservation(ts, price, volume)

            # 現在の行動マスクを取得
            masks = self.env.action_masks()

            # モデルによる行動予測
            action, _states = self.model.predict(obs, masks=masks)

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
            for key, value in dict_technicals.items():
                self.dict_list_tech[key].append(value)

            # -----------------------------------------------------------------
            # アクションによる環境の状態更新
            # 【注意】 リアルタイム用環境では step メソッドで観測値は返されない
            # -----------------------------------------------------------------
            reward, terminated, truncated, info = self.env.step(action)
            if terminated:
                self.logger.info("terminated フラグが立ちました。")
                self.done = True
                # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # 🧿 取引終了
                self.completedTrading.emit()
                # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            elif truncated:
                self.logger.info("truncated フラグが立ちました。")
                self.done = True
                # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # 🧿 取引終了
                self.completedTrading.emit()
                # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    @Slot()
    def forceRepay(self) -> None:
        self.env.forceRepay()

    @Slot()
    def resetEnv(self) -> None:
        # 環境のリセット
        self.obs, _ = self.env.reset()
        self.done = False
        self._is_stopping = False

        list_colname = ["Timestamp", "Price", "Volume"]
        self.list_obs_label = self.env.getObsList()
        self.model.updateObs(self.list_obs_label)
        list_colname.extend(self.list_obs_label)
        dict_colname = {colname: [] for colname in list_colname}
        self.df_obs = pd.DataFrame(dict_colname)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 環境のリセット完了を通知
        self.completedResetEnv.emit()
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    @Slot(str)
    def saveTechnicals(self, path_csv: str) -> None:
        """
        テクニカル指標を CSV ファイルに保存
        :param path_csv: 保存先のファイルパス
        """
        try:
            df = pd.DataFrame(self.dict_list_tech)
            # インデックスを日付形式に変換
            df.index = [pd.to_datetime(conv_datetime_from_timestamp(ts)) for ts in df["ts"]]
            # 指定されたパスにデータフレームを CSV 形式で保存
            df.to_csv(path_csv)
            self.logger.info(f"{__name__}: {path_csv} を保存しました。")
        except Exception as e:
            self.logger.error(f"{__name__}: テクニカル指標の保存に失敗しました: {e}")

    @Slot()
    def cleanup(self) -> None:
        """
        スレッド終了前のクリーンアップ処理
        Trader.closeEvent から呼び出される想定（オプション）
        """
        self.logger.info(f"{__name__}: ワーカーのクリーンアップを開始します。")
        self._is_stopping = True

        # 必要に応じてリソースの解放処理を追加
        # 例：self.env.close() などがあれば呼び出す

        self.logger.info(f"{__name__}: ワーカーのクリーンアップが完了しました。")


class CronAgent:
    """
    cron で実行できる GUI を利用しないエージェント
    """

    def __init__(self, code: str, dict_ts: dict):
        self.logger = logging.getLogger(__name__)
        self.code = code
        self.ts_end = dict_ts["end"]

        # モデルのインスタンス
        self.df_obs = None
        self.list_obs_label = list()
        self.model = AlgoTrade()

        self.list_ts = list()
        self.list_obs = list()

        # ポジション・マネージャ
        self.posman = PositionManager()
        self.posman.initPosition([code])

        # 環境クラス
        self.env: TradingEnv | None = None

        # 取引内容（＋テクニカル指標）
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
        self.list_ts.append(ts)
        self.list_obs.append(obs)

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

    def getObservations(self) -> pd.DataFrame:
        df = pd.DataFrame(np.array(self.list_obs))
        df.columns = self.list_obs_label
        df.index = [pd.to_datetime(conv_datetime_from_timestamp(ts)) for ts in self.list_ts]
        return df

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
        # self.list_obs_label.clear()
        # self.list_obs_label.extend(self.env.getObsList())
        self.list_obs_label = None
        self.list_obs_label = self.env.getObsList()
        self.model.updateObs(self.list_obs_label)

        list_colname.extend(self.list_obs_label)
        dict_colname = dict()
        for colname in list_colname:
            dict_colname[colname] = []
        self.df_obs = pd.DataFrame(dict_colname)
