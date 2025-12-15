import logging

import numpy as np
import pandas as pd
from PySide6.QtCore import Signal, QThread, Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QMainWindow

from funcs.ios import load_setting
from funcs.models import get_trained_ppo_model_path
from widgets.charts import TrendChart
from modules.dock import DockTrader
from modules.agent import WorkerAgent
from structs.app_enum import ActionType, PositionType
from structs.res import AppRes


class Trader(QMainWindow):
    notifyAutoPilotStatus = Signal(bool)
    sendTradeData = Signal(float, float, float)
    requestResetEnv = Signal()

    def __init__(self, res: AppRes, code: str):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.res = res
        self.code = code

        # タイムスタンプへ時差を加算・減算用（Asia/Tokyo)
        self.tz = 9. * 60 * 60

        #######################################################################
        # データ点を追加する毎に再描画するので、あらかじめ配列を確保し、
        # スライスでデータを渡すようにして、なるべく描画以外の処理を減らす。
        #

        # 最大データ点数（昼休みを除く 9:00 - 15:30 まで　1 秒間隔のデータ数）
        self.max_data_points = 19800

        # データ領域の確保
        self.x_data = np.empty(self.max_data_points, dtype=pd.Timestamp)
        self.y_data = np.empty(self.max_data_points, dtype=np.float64)
        self.v_data = np.empty(self.max_data_points, dtype=np.float64)

        # データ点用のカウンター
        self.count_data = 0

        #
        #######################################################################

        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        #  UI
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # ウィンドウのサイズ制約
        # self.setMinimumWidth(1200)
        # self.setFixedHeight(300)

        # ---------------------------------------------------------------------
        # 右側のドック
        # ---------------------------------------------------------------------
        self.dock = dock = DockTrader(res, code)
        self.dock.option.changedAutoPilotStatus.connect(self.changedAutoPilotStatus)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

        # ---------------------------------------------------------------------
        # チャートインスタンス (FigureCanvas)
        # ---------------------------------------------------------------------
        self.chart = chart = TrendChart(res)
        self.setCentralWidget(chart)

        # 最新の株価
        self.latest_point, = self.chart.ax.plot(
            [], [],
            marker='x',
            markersize=7,
            color='#fc8'
        )

        # トレンドライン（株価）
        self.trend_line, = self.chart.ax.plot(
            [], [],
            color='lightgray',
            linewidth=0.5
        )

        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # 強化学習モデル用スレッド
        self.thread = QThread(self)
        # 学習済みモデルのパス
        path_model = get_trained_ppo_model_path(res, code)
        # AutoPilot フラグ
        flag_autopilot = self.dock.option.isAutoPilotEnabled()
        # 銘柄コード別設定ファイルの取得
        dict_setting = load_setting(res, code)

        # ワーカースレッドの生成
        self.worker = WorkerAgent(flag_autopilot, code, dict_setting)
        self.worker.moveToThread(self.thread)

        # メインスレッドのシグナル処理 → ワーカースレッドのスロットへ
        self.notifyAutoPilotStatus.connect(self.worker.setAutoPilotStatus)
        self.requestResetEnv.connect(self.worker.resetEnv)
        self.sendTradeData.connect(self.worker.addData)

        # ワーカースレッドからのシグナル処理 → メインスレッドのスロットへ
        self.worker.completedResetEnv.connect(self.reset_env_completed)
        self.worker.notifyAction.connect(self.on_action)

        # スレッドの開始
        self.thread.start()
        # エージェント環境のリセット → リセット終了で処理開始
        self.requestResetEnv.emit()
        #
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_

    def changedAutoPilotStatus(self, state: bool):
        self.notifyAutoPilotStatus.emit(state)

    def closeEvent(self, event: QCloseEvent):
        # self.worker.stop()
        # self.thread.quit()
        # self.thread.wait()
        event.accept()

    def getTimePrice(self) -> pd.DataFrame:
        """
        保持している時刻、株価情報をデータフレームで返す。
        :return:
        """
        # タイムスタンプ の Time 列は self.tz を考慮
        return pd.DataFrame({
            "Time": [t.timestamp() - self.tz for t in self.x_data[0: self.count_data]],
            "Price": self.y_data[0: self.count_data],
            "Volume": self.v_data[0: self.count_data],
        })

    def on_action(self, action: int, position: PositionType):
        action_enum = ActionType(action)
        if action_enum == ActionType.BUY:
            if position == PositionType.NONE:
                # 建玉がなければ買建
                self.dock.doBuy()
            elif position == PositionType.SHORT:
                # 売建（ショート）であれば（買って）返済
                self.dock.doRepay()
            else:
                self.logger.error(f"{__name__}: trade rule violation!")
        elif action_enum == ActionType.SELL:
            if position == PositionType.NONE:
                # 建玉がなければ売建
                self.dock.doSell()
            elif position == PositionType.LONG:
                # 買建（ロング）であれば（売って）返済
                self.dock.doRepay()
            else:
                self.logger.error(f"{__name__}: trade rule violation!")
        elif action_enum == ActionType.HOLD:
            pass
        else:
            self.logger.error(f"{__name__}: unknown action type {action_enum}!")

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
        self.chart.ax.axhline(y=price_close, color="red", linewidth=0.75)

    def setTradeData(self, ts: float, price: float, volume: float):
        """
        ティックデータの取得
        :param ts:
        :param price:
        :param volume:
        :return:
        """
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 ティックデータを送るシグナル
        self.sendTradeData.emit(ts, price, volume)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # ---------------------------------------------------------------------
        # ts（タイムスタンプ）から、Matplotlib 用の値＝タイムスタンプ（時差込み）に変換
        # ---------------------------------------------------------------------
        x = pd.Timestamp(ts + self.tz, unit='s')

        # ---------------------------------------------------------------------
        # 最新の株価
        # ---------------------------------------------------------------------
        self.latest_point.set_xdata([x])
        self.latest_point.set_ydata([price])

        # ---------------------------------------------------------------------
        # 配列に保持
        # ---------------------------------------------------------------------
        self.x_data[self.count_data] = x
        self.y_data[self.count_data] = price
        self.v_data[self.count_data] = volume
        self.count_data += 1
        # ---------------------------------------------------------------------
        # 株価トレンド線
        # ---------------------------------------------------------------------
        self.trend_line.set_xdata(self.x_data[0:self.count_data])
        self.trend_line.set_ydata(self.y_data[0:self.count_data])

        # 再描画
        self.chart.reDraw()

    def setTimeAxisRange(self, ts_start, ts_end):
        """
        x軸のレンジ
        固定レンジで使いたいため。
        ただし、前場と後場で分ける機能を検討する余地はアリ
        :param ts_start:
        :param ts_end:
        :return:
        """
        pad_left = 5. * 60  # チャート左側の余白（５分）
        dt_start = pd.Timestamp(ts_start + self.tz - pad_left, unit='s')
        dt_end = pd.Timestamp(ts_end + self.tz, unit='s')
        self.chart.ax.set_xlim(dt_start, dt_end)

    def setChartTitle(self, title: str):
        """
        チャートのタイトルを設定
        :param title:
        :return:
        """
        self.chart.setTitle(title)
