import logging

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow

from structs.res import AppRes
from widgets.docks import DockTrader
from widgets.graph import TrendGraph


class Trader(QMainWindow):
    def __init__(self, res: AppRes):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.res = res

        # 最大データ点数（昼休みを除く 9:00 - 15:30 まで　1 秒間隔のデータ数）
        self.max_data_points = 19800

        # カウンター
        self.counter = 0

        # データ領域の確保
        self.data_x = np.empty(self.max_data_points, dtype=np.float64)
        self.data_y = np.empty(self.max_data_points, dtype=np.float64)
        self.bull_y = np.empty(self.max_data_points, dtype=np.float64)
        self.bear_y = np.empty(self.max_data_points, dtype=np.float64)

        # 右側のドック
        self.dock = dock = DockTrader(res)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

        # PyQtGraph インスタンス
        self.chart = chart = TrendGraph()
        self.setCentralWidget(chart)

        # 株価トレンドライン
        self.trend_line: pg.PlotDataItem = chart.plot(pen=pg.mkPen(width=1))

        # 最新株価
        # self.point_latest: pg.PlotDataItem = chart.plot(symbol='o', symbolSize=5, pxMode=True)
        self.point_latest = pg.ScatterPlotItem(
            size=5,  # 例として少し小さめに
            # pen=pg.mkPen(color=(255, 165, 0), width=1), # 緑色の境界線
            pen=None,
            brush=pg.mkBrush(color=(255, 165, 0)),
            symbol='o',  # 丸い点
            pxMode=True,  # サイズをピクセル単位で固定
            antialias=False  # アンチエイリアスをオフにすると少し速くなる可能性も
        )
        chart.addItem(self.point_latest)

        # 前日終値
        self.lastclose_line: pg.InfiniteLine | None = None

    def appendData(self, x, y):
        self.data_x[self.counter] = x
        self.data_y[self.counter] = y
        self.counter += 1

        self.trend_line.setData(
            self.data_x[0: self.counter], self.data_y[0:self.counter]
        )
        self.point_latest.setData([x], [y])

    def setTimeRange(self, ts_start, ts_end):
        self.chart.setXRange(ts_start, ts_end)

    def setTitle(self, title: str):
        self.chart.setTitle(title)
