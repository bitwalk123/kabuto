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

    def setTimeRange(self, ts_start, ts_end):
        self.chart.setXRange(ts_start, ts_end)

    def setTitle(self, title: str):
        self.chart.setTitle(title)
