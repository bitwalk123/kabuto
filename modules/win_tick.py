import logging

import numpy as np
import pandas as pd
from PySide6.QtCore import (
    QObject,
    QThread,
    Signal,
    Slot,
)

from funcs.ios import get_excel_sheet
from structs.res import AppRes
from widgets.charts import ChartNavigation, TickChart
from widgets.containers import MainWindow
from widgets.statusbars import StatusBar


class TickLoader(QObject):
    finished = Signal(pd.DataFrame)  # 読み込み完了時にデータを送る

    def __init__(self, path_excel: str, code: str):
        super().__init__()
        self.path_excel = path_excel
        self.code = code

    @Slot()
    def run(self):
        df = get_excel_sheet(self.path_excel, self.code)
        self.finished.emit(df)


class WinTick(MainWindow):
    def __init__(self, res: AppRes):
        super().__init__()
        self.logger = logging.getLogger(__name__)  # モジュール固有のロガーを取得
        self.res = res
        self.title: str = ""

        self.thread = None
        self.worker = None

        #######################################################################
        # データ点を追加する毎に再描画するので、あらかじめ配列を確保し、
        # スライスでデータを渡すようにして、なるべく描画以外の処理を減らす。
        #

        # タイムスタンプへ時差を加算・減算用（Asia/Tokyo)
        self.tz = 9. * 60 * 60

        # 最大データ点数（昼休みを除く 9:00 - 15:30 まで　1 秒間隔のデータ数）
        self.max_data_points = 19800

        # データ領域の確保
        self.x_data = np.empty(self.max_data_points, dtype=pd.Timestamp)
        self.y_data = np.empty(self.max_data_points, dtype=np.float64)

        # データ点用のカウンター
        self.count_data = 0

        #
        #######################################################################

        # ---------------------------------------------------------------------
        # チャート
        # ---------------------------------------------------------------------
        self.chart = chart = TickChart(res)
        self.setCentralWidget(chart)

        # トレンドライン（株価）
        self.trend_line, = self.chart.ax.plot(
            [], [],
            color='C0',
            linewidth=0.5
        )

        # ---------------------------------------------------------------------
        # ステータスバー
        # ---------------------------------------------------------------------
        self.statusbar = statusbar = StatusBar(res)
        statusbar.setSizeGripEnabled(False)
        navbar = ChartNavigation(chart)
        statusbar.addWidget(navbar)
        self.setStatusBar(statusbar)

    def clearPlot(self):
        self.chart.clearPlot()
        # トレンドライン（株価）
        self.trend_line, = self.chart.ax.plot(
            [], [],
            color='C0',
            linewidth=0.5
        )
        # データ点用のカウンター
        self.count_data = 0

    def reDraw(self):
        self.chart.draw()

    def addData(self, ts: float, price: float):
        # ---------------------------------------------------------------------
        # ts（タイムスタンプ）から、Matplotlib 用の値＝タイムスタンプ（時差込み）に変換
        # ---------------------------------------------------------------------
        x = pd.Timestamp(ts + self.tz, unit='s')

        # ---------------------------------------------------------------------
        # 配列に保持
        # ---------------------------------------------------------------------
        self.x_data[self.count_data] = x
        self.y_data[self.count_data] = price
        self.count_data += 1

        # ---------------------------------------------------------------------
        # 株価トレンド線
        # ---------------------------------------------------------------------
        self.trend_line.set_xdata(self.x_data[0:self.count_data])
        self.trend_line.set_ydata(self.y_data[0:self.count_data])

        self.chart.ax.relim()
        self.chart.ax.autoscale_view(scalex=False, scaley=True)

        # 再描画
        self.reDraw()

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

    def updateChart(self, path_excel: str, code: str, title: str):
        """
        チャートの更新
        Args:
            path_excel: Excel ファイル（フルパス）
            code: 銘柄コード
            title: チャートのタイトル

        Returns:

        """
        self.title = title

        self.thread = thread = QThread()
        self.worker = worker = TickLoader(path_excel, code)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.finished.connect(self.on_update_chart)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        # スレッドの開始
        thread.start()

    @Slot(pd.DataFrame)
    def on_update_chart(self, df: pd.DataFrame):
        self.chart.updateData(df, self.title)
