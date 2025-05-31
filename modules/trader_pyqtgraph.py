import logging

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow

from modules.psar import RealtimePSAR
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

        # データ領域の確保
        self.x_data = np.empty(self.max_data_points, dtype=np.float64)
        self.y_data = np.empty(self.max_data_points, dtype=np.float64)
        # データ点用のカウンター
        self.counter_data = 0

        # Parabolic SAR
        self.psar = RealtimePSAR()

        # bull（情報トレンド）
        self.x_bull = np.empty(self.max_data_points, dtype=np.float64)
        self.y_bull = np.empty(self.max_data_points, dtype=np.float64)
        # bull 用のカウンター
        self.counter_bull = 0

        # bear（下降トレンド）
        self.x_bear = np.empty(self.max_data_points, dtype=np.float64)
        self.y_bear = np.empty(self.max_data_points, dtype=np.float64)
        # bear 用のカウンター
        self.counter_bear = 0

        # 右側のドック
        self.dock = dock = DockTrader(res)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

        # PyQtGraph インスタンス
        self.chart = chart = TrendGraph()
        self.setCentralWidget(chart)

        # 株価トレンドライン
        self.trend_line: pg.PlotDataItem = chart.plot(pen=pg.mkPen(width=1))

        # 最新株価
        self.point_latest = pg.ScatterPlotItem(
            size=6,
            pen=None,
            brush=pg.mkBrush(color=(255, 165, 0)),
            symbol='o',
            pxMode=True,  # サイズをピクセル単位で固定
            antialias=False  # アンチエイリアスをオフにすると少し速くなる可能性も
        )
        chart.addItem(self.point_latest)

        # 前日終値
        self.lastclose_line: pg.InfiniteLine | None = None

        # bull（上昇トレンド）
        self.trend_bull = pg.ScatterPlotItem(
            size=3,
            pen=pg.mkPen(color=(255, 0, 255)),
            brush=None,
            symbol='o',
            pxMode=True,  # サイズをピクセル単位で固定
            antialias=False  # アンチエイリアスをオフにすると少し速くなる可能性も
        )
        chart.addItem(self.trend_bull)

        # bear（下降トレンド）
        self.trend_bear = pg.ScatterPlotItem(
            size=3,
            pen=pg.mkPen(color=(0, 255, 255)),
            brush=None,
            symbol='o',
            pxMode=True,  # サイズをピクセル単位で固定
            antialias=False  # アンチエイリアスをオフにすると少し速くなる可能性も
        )
        chart.addItem(self.trend_bear)

    def addData(self, x: np.float64, y: np.float64):
        self.x_data[self.counter_data] = x
        self.y_data[self.counter_data] = y
        self.counter_data += 1

        self.trend_line.setData(
            self.x_data[0: self.counter_data], self.y_data[0:self.counter_data]
        )
        self.point_latest.setData([x], [y])

        # Parabolic SAR
        ret = self.psar.add(y)
        y_psar = ret.psar
        if 0 < ret.trend:
            self.x_bull[self.counter_bull] = x
            self.y_bull[self.counter_bull] = y_psar
            self.counter_bull += 1
            self.trend_bull.setData(
                self.x_bull[0: self.counter_bull], self.y_bull[0:self.counter_bull]
            )
        elif ret.trend < 0:
            self.x_bear[self.counter_bear] = x
            self.y_bear[self.counter_bear] = y_psar
            self.counter_bear += 1
            self.trend_bear.setData(
                self.x_bear[0: self.counter_bear], self.y_bear[0:self.counter_bear]
            )

    def setTimeRange(self, ts_start, ts_end):
        self.chart.setXRange(ts_start, ts_end)

    def setTitle(self, title: str):
        self.chart.setTitle(title)
