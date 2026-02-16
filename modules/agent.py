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
from structs.app_enum import ActionType, PositionType


class WorkerAgent(QObject):
    """
    強化学習を利用せずに、アルゴリズムのみのエージェント
    （リアルタイム用）
    """
    BASE_COLUMNS = ["Timestamp", "Price", "Volume"]

    # シグナル
    completedResetEnv = Signal()
    completedTrading = Signal()
    notifyAction = Signal(int, PositionType)  # 売買アクションを通知
    sendTechnicals = Signal(dict)

    def __init__(self, code: str, dict_param: dict[str, Any]) -> None:
        super().__init__()
        self.logger = logging.getLogger(__name__)

        self.obs: np.ndarray | None = None
        self.done: bool = False
        self.df_obs: pd.DataFrame | None = None
        self._is_stopping: bool = False  # 終了フラグを追加

        # 学習環境の取得
        self.env: TradingEnv = TradingEnv(code, dict_param)

        # モデルに渡す観測値のリスト
        self.list_obs_label: list[str] = []
        # モデルのインスタンス
        self.model: AlgoTrade = AlgoTrade()

        # 取引内容（＋テクニカル指標）
        self.dict_list_tech = defaultdict(list)

    @Slot(float, float, float)
    def addData(self, ts: float, price: float, volume: float) -> None:
        # 終了処理中はデータを処理しない
        if self._is_stopping or self.done:
            return

        # ティックデータから観測値を取得
        obs, dict_technicals = self.env.getObservation(ts, price, volume)

        # 現在の行動マスクを取得
        masks: np.ndarray = self.env.action_masks()

        # モデルによる行動予測
        action, _states = self.model.predict(obs, masks=masks)
        position: PositionType = self.env.getCurrentPosition()
        if ActionType(action) != ActionType.HOLD:
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # 🧿 売買アクションを通知するシグナル（HOLD の時は通知しない）
            self.notifyAction.emit(action, position)
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # プロット用テクニカル指標
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 テクニカル指標を通知するシグナル
        self.sendTechnicals.emit(dict_technicals)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        for key, value in dict_technicals.items():
            self.dict_list_tech[key].append(value)

        # ---------------------------------------------------------------------
        # アクションによる環境の状態更新
        # 【注意】 リアルタイム用環境では step メソッドで観測値は返されない
        # ---------------------------------------------------------------------
        reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            flag_name = "terminated" if terminated else "truncated"
            self.logger.info(f"{flag_name} フラグが立ちました。")
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

        list_colname = self.BASE_COLUMNS.copy()
        self.list_obs_label = self.env.getObsList()
        self.model.updateObs(self.list_obs_label)
        list_colname.extend(self.list_obs_label)
        self.df_obs = pd.DataFrame({col: [] for col in list_colname})
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
            self.logger.info(f"{path_csv} を保存しました。")
        except (KeyError, ValueError, IOError) as e:
            self.logger.error(f"テクニカル指標の保存に失敗しました: {e}")

    @Slot()
    def cleanup(self) -> None:
        """
        スレッド終了前のクリーンアップ処理
        Trader.closeEvent から呼び出される想定（オプション）
        """
        self.logger.info(f"ワーカーのクリーンアップを開始します。")
        self._is_stopping = True

        # 必要に応じてリソースの解放処理を追加
        # 例：self.env.close() などがあれば呼び出す

        self.logger.info(f"ワーカーのクリーンアップが完了しました。")
