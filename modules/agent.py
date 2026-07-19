import logging
from collections import defaultdict
from typing import Any, DefaultDict

import numpy as np
import pandas as pd
from PySide6.QtCore import (
    QObject,
    Signal,
    Slot,
)

from funcs.plugin import get_model_instance
from funcs.tide import conv_datetime_from_timestamp
from modules.env import TradingEnv
from modules.posman import PositionManager
from structs.app_enum import ActionType, PositionType


class SimulationAgent:
    BASE_COLUMNS = ["Timestamp", "Price", "Volume"]

    def __init__(
            self,
            code: str,
            dict_setting: dict[str, Any]
    ) -> None:
        self.code = code
        self.obs: np.ndarray | None = None
        self.df_obs: pd.DataFrame | None = None

        # 学習環境の取得
        self.env: TradingEnv = TradingEnv(code, dict_setting)

        # モデルに渡す観測値のリスト
        self.list_obs_label: list[str] = []

        # モデルのインスタンス（とりあえずプラグイン化）
        # name_model = "default"
        name_model = "model_001"
        self.model = model = get_model_instance(name_model)
        model.setAutoPilot(True)

        # ポジションマネージャ
        self.posman = posman = PositionManager()
        posman.initPosition([code])

        # 取引内容（＋テクニカル指標）
        self.dict_list_tech: DefaultDict[str, list[Any]] = defaultdict(list)

    def run(self, df_technicals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        self.resetEnv()
        n_row = len(df_technicals)
        ts, price = (0.0, 0.0)
        for r in range(n_row):
            row = df_technicals.iloc[r]
            ts = row["Time"]
            price = row["Price"]
            volume = row["Volume"]
            dict_info = self.posman.getInfo(self.code, price)
            dict_technicals = self.addData(ts, price, volume, dict_info)
            # print(dict_technicals)
            # トレード後にまとめてデータフレームで出力するため
            for key, value in dict_technicals.items():
                self.dict_list_tech[key].append(value)

        # 建玉が残っていれば強制返済
        if self.posman.hasPosition(self.code):
            self.posman.closePosition(self.code, ts, price, "強制返済")

        df_technicals = pd.DataFrame(self.dict_list_tech)
        # インデックスを日付形式に変換
        df_technicals.index = [
            pd.to_datetime(conv_datetime_from_timestamp(ts)) for ts in df_technicals["ts"]
        ]
        df_transaction = self.posman.getTransactionResult()
        return df_technicals, df_transaction

    def addData(
            self,
            ts: float,
            price: float,
            volume: float,
            dict_info: dict
    ) -> dict[str, Any]:
        # ティックデータから観測値を取得
        obs, dict_technicals = self.env.getObservation(ts, price, volume, dict_info)

        # 現在の行動マスクを取得
        masks: np.ndarray = self.env.action_masks()

        # モデルによる行動予測
        action, states = self.model.predict(obs, action_masks=masks)
        action_enum = ActionType(action)

        # メイン・スレッドへ通知する発注アクション
        position: PositionType = self.env.getCurrentPosition()
        if "reason" in states:
            note = states["reason"]
        else:
            note = ""
        if position == PositionType.NONE:
            if action_enum == ActionType.BUY:
                self.posman.openPosition(self.code, ts, price, ActionType.BUY, note)
            elif action_enum == ActionType.SELL:
                self.posman.openPosition(self.code, ts, price, ActionType.SELL, note)
            else:
                pass
        elif position == PositionType.LONG:
            if action_enum == ActionType.BUY:
                pass
            elif action_enum == ActionType.SELL:
                self.posman.closePosition(self.code, ts, price, note)
            else:
                pass
        elif position == PositionType.SHORT:
            if action_enum == ActionType.BUY:
                self.posman.closePosition(self.code, ts, price, note)
            elif action_enum == ActionType.SELL:
                pass
            else:
                pass
        else:
            pass

        # メイン・スレッドへ通知するプロット用テクニカル指標
        # 🧿 テクニカル指標を通知するシグナル
        # self.sendTechnicals.emit(dict_technicals)
        return dict_technicals

    def resetEnv(self) -> None:
        # 環境のリセット
        self.obs, _ = self.env.reset()
        # self.done = False
        # self._is_stopping = False

        list_colname = self.BASE_COLUMNS.copy()
        self.list_obs_label = self.env.getObsList()
        self.model.updateObs(self.list_obs_label)
        list_colname.extend(self.list_obs_label)
        self.df_obs = pd.DataFrame({col: [] for col in list_colname})
        # 🧿 環境のリセット完了を通知
        # self.completedResetEnv.emit()


class WorkerAgent(QObject):
    """
    強化学習を利用せずに、アルゴリズムのみのエージェント
    （リアルタイム用）
    """
    BASE_COLUMNS = ["Timestamp", "Price", "Volume"]

    # シグナル
    completedResetEnv = Signal()
    completedTrading = Signal()
    notifyAction = Signal(int, PositionType, dict)  # 売買アクションを通知
    sendTechnicals = Signal(dict)

    def __init__(self, code: str, dict_setting: dict[str, Any]) -> None:
        super().__init__()
        self.logger = logging.getLogger(__name__)

        self.obs: np.ndarray | None = None
        self.df_obs: pd.DataFrame | None = None

        # 学習環境の取得
        self.env: TradingEnv = TradingEnv(code, dict_setting)

        # モデルに渡す観測値のリスト
        self.list_obs_label: list[str] = []

        # モデルのインスタンス（とりあえずプラグイン化）
        # name_model = "default"
        name_model = "model_001"
        self.model = get_model_instance(name_model)

        # 取引内容（＋テクニカル指標）
        self.dict_list_tech: DefaultDict[str, list[Any]] = defaultdict(list)

    @Slot(float, float, float, dict)
    def addData(self, ts: float, price: float, volume: float, dict_info: dict) -> None:
        # ティックデータから観測値を取得
        obs, dict_technicals = self.env.getObservation(ts, price, volume, dict_info)

        # 現在の行動マスクを取得
        masks: np.ndarray = self.env.action_masks()

        # モデルによる行動予測
        action, states = self.model.predict(obs, action_masks=masks)

        # メイン・スレッドへ通知する発注アクション
        position: PositionType = self.env.getCurrentPosition()
        if ActionType(action) != ActionType.HOLD:
            # 🧿 売買アクションを通知するシグナル（HOLD の時は通知しない）
            self.notifyAction.emit(action, position, states)

        # メイン・スレッドへ通知するプロット用テクニカル指標
        # 🧿 テクニカル指標を通知するシグナル
        self.sendTechnicals.emit(dict_technicals)
        # トレード後にまとめてデータフレームで出力するため
        for key, value in dict_technicals.items():
            self.dict_list_tech[key].append(value)

    @Slot()
    def cleanup(self) -> None:
        """
        スレッド終了前のクリーンアップ処理
        Trader.closeEvent から呼び出される想定（オプション）
        """
        self.logger.info(f"ワーカーのクリーンアップを開始します。")

        # 必要に応じてリソースの解放処理を追加
        # 例：self.env.close() などがあれば呼び出す

        self.logger.info(f"ワーカーのクリーンアップが完了しました。")

    @Slot()
    def forceRepay(self) -> None:
        """
        建玉返済の強制処理通知
        :return:
        """
        self.env.forceRepay()

    @Slot()
    def resetEnv(self) -> None:
        # 環境のリセット
        self.obs, _ = self.env.reset()
        # self.done = False
        # self._is_stopping = False

        list_colname = self.BASE_COLUMNS.copy()
        self.list_obs_label = self.env.getObsList()
        self.model.updateObs(self.list_obs_label)
        list_colname.extend(self.list_obs_label)
        self.df_obs = pd.DataFrame({col: [] for col in list_colname})
        # 🧿 環境のリセット完了を通知
        self.completedResetEnv.emit()

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
            self.logger.info(f"{path_csv} を保存しました。")
        except (KeyError, ValueError, IOError) as e:
            self.logger.error(f"テクニカル指標の保存に失敗しました: {e}")

    @Slot(bool)
    def setAutoPilot(self, flag: bool):
        self.model.setAutoPilot(flag)

    @Slot(bool)
    def updateStateCross(self, state: bool):
        state_new = self.env.s.setStatusCross(state)
        self.logger.info(f"Cross 返済が {state_new} に変更されました。")
