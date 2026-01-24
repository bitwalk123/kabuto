import logging
import os

import pandas as pd
from PySide6.QtCore import (
    QThread,
    Qt,
    Signal,
)
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QMainWindow

from funcs.setting import load_setting
from modules.agent import WorkerAgentRT
from modules.dock import DockTrader
from structs.app_enum import ActionType, PositionType
from structs.res import AppRes
from widgets.graphs import TrendGraph


class Trader(QMainWindow):
    notifyAutoPilotStatus = Signal(bool)
    sendTradeData = Signal(float, float, float)
    requestResetEnv = Signal()

    # --- 状態遷移表 ---
    ACTION_DISPATCH = {
        (ActionType.BUY, PositionType.NONE): "doBuy",  # 建玉がなければ買建
        (ActionType.BUY, PositionType.SHORT): "doRepay",  # 売建（ショート）であれば（買って）返済
        (ActionType.SELL, PositionType.NONE): "doSell",  # 建玉がなければ売建
        (ActionType.SELL, PositionType.LONG): "doRepay",  # 買建（ロング）であれば（売って）返済
        # HOLD は何もしないので載せない
    }

    def __init__(self, res: AppRes, code: str, dict_ts: dict):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.res = res
        self.code = code
        self.dict_ts = dict_ts

        # タイムスタンプへ時差を加算・減算用（Asia/Tokyo)
        # self.tz = 9. * 60 * 60

        # ティックデータ
        self.list_x = list()
        self.list_y = list()
        self.list_v = list()
        # テクニカル指標
        self.list_ts = list()  # self.list_x と同一になってしまうかもしれない
        self.list_ma_1 = list()
        self.list_ma_2 = list()

        # 銘柄コード別設定ファイルの取得
        dict_setting = load_setting(res, code)

        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        #  UI
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_

        # ---------------------------------------------------------------------
        # 右側のドック
        # ---------------------------------------------------------------------
        self.dock = dock = DockTrader(res, code)
        self.dock.option.changedAutoPilotStatus.connect(self.changedAutoPilotStatus)
        self.dock.clickedSave.connect(self.on_save)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

        # ---------------------------------------------------------------------
        # チャートインスタンス
        # ---------------------------------------------------------------------
        self.trend = trend = TrendGraph(res, dict_ts, dict_setting)
        self.setCentralWidget(trend)

        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # 強化学習モデル用スレッド
        self.thread = QThread(self)

        # 学習済みモデルのパス
        # path_model = get_trained_ppo_model_path(res, code)

        # AutoPilot フラグ
        flag_autopilot = self.dock.option.isAutoPilotEnabled()

        # ワーカースレッドの生成
        self.worker = WorkerAgentRT(flag_autopilot, code, dict_setting)
        self.worker.moveToThread(self.thread)

        # メインスレッドのシグナル処理 → ワーカースレッドのスロットへ
        self.notifyAutoPilotStatus.connect(self.worker.setAutoPilotStatus)
        self.requestResetEnv.connect(self.worker.resetEnv)
        self.sendTradeData.connect(self.worker.addData)

        # ワーカースレッドからのシグナル処理 → メインスレッドのスロットへ
        self.worker.completedResetEnv.connect(self.reset_env_completed)
        self.worker.completedTrading.connect(self.on_trading_completed)
        self.worker.notifyAction.connect(self.on_action)
        self.worker.sendTechnicals.connect(self.on_technicals)

        # スレッドの開始
        self.thread.start()
        # エージェント環境のリセット → リセット終了で処理開始
        self.requestResetEnv.emit()
        #
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_

    def changedAutoPilotStatus(self, state: bool):
        self.notifyAutoPilotStatus.emit(state)

    def closeEvent(self, event: QCloseEvent):
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait()

        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None

        if self.thread is not None:
            self.thread.deleteLater()
            self.thread = None

        self.logger.info(f"{__name__}: スレッドを終了しました。")
        event.accept()

    def getTimePrice(self) -> pd.DataFrame:
        """
        保持している時刻、株価情報をデータフレームで返す。
        :return:
        """
        return pd.DataFrame({
            "Time": self.list_x,
            "Price": self.list_y,
            "Volume": self.list_v,
        })

    def on_action(self, action: int, position: PositionType):
        action_enum = ActionType(action)

        # HOLD は即 return
        if action_enum == ActionType.HOLD:
            return

        method_name = self.ACTION_DISPATCH.get((action_enum, position))
        if method_name is None:
            self.logger.error(
                f"{__name__}: trade rule violation! action={action_enum}, pos={position}"
            )
            return

        # dock のメソッドを取得して実行
        getattr(self.dock, method_name)()

    def on_save(self):
        file_img = f"{self.dict_ts['datetime_str']}_{self.code}_trend.png"
        path_img = os.path.join(self.res.dir_output, file_img)
        self.trend.save(path_img)

    def on_technicals(self, dict_technicals: dict):
        self.list_ts.append(dict_technicals["ts"])
        self.list_ma_1.append(dict_technicals["ma1"])
        self.list_ma_2.append(dict_technicals["ma2"])
        # テクニカル指標
        self.trend.setTechnicals(
            self.list_ts,
            self.list_ma_1,
            self.list_ma_2
        )

    def on_trading_completed(self):
        self.logger.info("取引が終了しました。")

    def reset_env_completed(self):
        """
        環境をリセット済
        :return:
        """
        msg = f"{__name__}: 銘柄コード {self.code} 用の環境がリセットされました。"
        self.logger.info(msg)

    def setLastCloseLine(self, price_close: float):
        """
        前日終値ラインの描画
        :param price_close:
        :return:
        """
        self.trend.ax.axhline(y=price_close, color="red", linewidth=0.75)

    def setTradeData(self, ts: float, price: float, volume: float):
        """
        ティックデータの取得
        :param ts:
        :param price:
        :param volume:
        :return:
        """
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # ティックデータを送るシグナル
        self.sendTradeData.emit(ts, price, volume)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # リストに保持
        self.list_x.append(ts)
        self.list_y.append(price)
        self.list_v.append(volume)

        # 株価トレンド線
        self.trend.setLine(self.list_x, self.list_y)

    def setTimeAxisRange(self, ts_start, ts_end):
        """
        x軸のレンジ
        固定レンジで使いたいため。
        ただし、前場と後場で分ける機能を検討する余地はアリ
        :param ts_start:
        :param ts_end:
        :return:
        """
        self.trend.setXRange(ts_start, ts_end)

    def setChartTitle(self, title: str):
        """
        チャートのタイトルを設定
        :param title:
        :return:
        """
        self.trend.setTrendTitle(title)
