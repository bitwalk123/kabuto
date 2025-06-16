import logging

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow

from structs.res import AppRes
from widgets.containers import Widget
from widgets.docks import DockTrader
from widgets.graph import TrendGraph, TrendGraph2
from widgets.layouts import VBoxLayout


class Trader(QMainWindow):
    def __init__(self, res: AppRes, ticker: str):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.res = res
        self.ticker = ticker

        #######################################################################
        # PyQtGraph では、データ点を追加する毎に再描画するので、あらかじめ配列を確保し、
        # スライスでデータを渡すようにして、なるべく描画以外の処理を減らす。

        # 最大データ点数（昼休みを除く 9:00 - 15:30 まで　1 秒間隔のデータ数）
        self.max_data_points = 19800

        # データ領域の確保
        self.x_data = np.empty(self.max_data_points, dtype=np.float64)
        self.y_data = np.empty(self.max_data_points, dtype=np.float64)
        # データ点用のカウンター
        self.counter_data = 0

        # bull（上昇トレンド）
        self.x_bull = np.empty(self.max_data_points, dtype=np.float64)
        self.y_bull = np.empty(self.max_data_points, dtype=np.float64)
        # bull 用のカウンター
        self.counter_bull = 0

        # bear（下降トレンド）
        self.x_bear = np.empty(self.max_data_points, dtype=np.float64)
        self.y_bear = np.empty(self.max_data_points, dtype=np.float64)
        # bear 用のカウンター
        self.counter_bear = 0

        # MR
        self.x_mr = np.empty(self.max_data_points, dtype=np.float64)
        self.y_mr = np.empty(self.max_data_points, dtype=np.float64)
        # MR 用のカウンター
        self.counter_mr = 0
        #
        #######################################################################

        # 右側のドック
        self.dock = dock = DockTrader(res, ticker)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

        base = Widget()
        self.setCentralWidget(base)
        layout = VBoxLayout()
        base.setLayout(layout)

        # ---------------------
        # PyQtGraph インスタンス
        # ---------------------
        self.chart = chart = TrendGraph()
        xaxis = chart.getAxis('bottom')
        xaxis.setStyle(showValues=False)
        xaxis.showLabel(False)
        layout.addWidget(chart)

        # 株価トレンドライン
        self.trend_line: pg.PlotDataItem = chart.plot(pen=pg.mkPen(width=0.5))

        # 最新株価の点
        self.point_latest = pg.ScatterPlotItem(
            size=10,
            pen=None,
            brush=pg.mkBrush(color=(255, 165, 0)),
            symbol='x',
            pxMode=True,  # サイズをピクセル単位で固定
            antialias=False  # アンチエイリアスをオフにすると少し速くなる可能性も
        )
        chart.addItem(self.point_latest)

        # 前日終値
        self.lastclose_line: pg.InfiniteLine | None = None

        # bull（Parabolic SAR 上昇トレンド）
        self.trend_bull = pg.ScatterPlotItem(
            size=1,
            pen=pg.mkPen(color=(255, 0, 255)),
            brush=None,
            symbol='o',
            pxMode=True,  # サイズをピクセル単位で固定
            antialias=False  # アンチエイリアスをオフにすると少し速くなる可能性も
        )
        chart.addItem(self.trend_bull)

        # bear（Parabolic SAR 下降トレンド）
        self.trend_bear = pg.ScatterPlotItem(
            size=1,
            pen=pg.mkPen(color=(0, 255, 255)),
            brush=None,
            symbol='o',
            pxMode=True,  # サイズをピクセル単位で固定
            antialias=False  # アンチエイリアスをオフにすると少し速くなる可能性も
        )
        chart.addItem(self.trend_bear)

        self.psar_latest = pg.ScatterPlotItem(
            size=8,
            pen=pg.mkPen(color=(0, 0, 0)),
            brush=None,
            symbol='o',
            pxMode=True,  # サイズをピクセル単位で固定
            antialias=False  # アンチエイリアスをオフにすると少し速くなる可能性も
        )
        chart.addItem(self.psar_latest)

        # ----------------------
        # PyQtGraph インスタンス２
        # ----------------------
        self.chart2 = chart2 = TrendGraph2()
        layout.addWidget(chart2)
        # x軸を chart とリンク
        chart2.setXLink(chart)

        # MR
        self.trend_index: pg.PlotDataItem = chart2.plot(pen=pg.mkPen(color=(128, 128, 0), width=1))

    def getTimePrice(self) -> pd.DataFrame:
        """
        保持している時刻、株価情報をデータフレームで返す。
        :return:
        """
        return pd.DataFrame({
            "Time": self.x_data[0: self.counter_data],
            "Price": self.y_data[0: self.counter_data]
        })

    def setLastCloseLine(self, price_close: float):
        """
        前日終値ラインの描画
        :param price_close:
        :return:
        """
        self.lastclose_line = pg.InfiniteLine(
            pos=price_close,
            angle=0,
            pen=pg.mkPen(color=(255, 0, 0), width=1)
        )
        self.chart.addItem(self.lastclose_line)

    def setIndex(self, x: np.float64, y: np.float64):
        self.x_mr[self.counter_mr] = x
        self.y_mr[self.counter_mr] = y
        self.counter_mr += 1

        self.trend_index.setData(
            self.x_mr[0: self.counter_mr], self.y_mr[0:self.counter_mr]
        )

    def setPSAR(self, trend: int, x: float, y: float, epupd: int):
        if 0 < trend:
            self.x_bull[self.counter_bull] = x
            self.y_bull[self.counter_bull] = y
            self.counter_bull += 1
            self.trend_bull.setData(
                self.x_bull[0: self.counter_bull], self.y_bull[0:self.counter_bull]
            )
            self.psar_latest.setPen(pg.mkPen(color=(255, 0, 255)))
            self.psar_latest.setBrush(pg.mkBrush(color=(255, 64, 255)))
            self.psar_latest.setData([x], [y])
        elif trend < 0:
            self.x_bear[self.counter_bear] = x
            self.y_bear[self.counter_bear] = y
            self.counter_bear += 1
            self.trend_bear.setData(
                self.x_bear[0: self.counter_bear], self.y_bear[0:self.counter_bear]
            )
            self.psar_latest.setPen(pg.mkPen(color=(0, 255, 255)))
            self.psar_latest.setBrush(pg.mkBrush(color=(64, 255, 255)))
            self.psar_latest.setData([x], [y])
        else:
            self.psar_latest.setPen(pg.mkPen(None))
            self.psar_latest.setBrush(pg.mkBrush(None))
            self.psar_latest.setData([x], [y])

        # トレンドの向きをドックに設定
        self.dock.setTrend(trend, epupd)

    def setTimePrice(self, x: np.float64, y: np.float64):
        """
        時刻、株価の追加
        あらかじめ確保しておいた配列を用い、
        カウンタで位置を管理してスライスで PyQtGraoh へ渡す
        :param x:
        :param y:
        :return:
        """
        self.x_data[self.counter_data] = x
        self.y_data[self.counter_data] = y
        self.counter_data += 1

        self.trend_line.setData(
            self.x_data[0: self.counter_data], self.y_data[0:self.counter_data]
        )
        self.point_latest.setData([x], [y])

        # 株価表示の更新
        self.dock.setPrice(y)

    def setTimeRange(self, ts_start, ts_end):
        """
        x軸のレンジ
        固定レンジで使いたいため。
        ただし、前場と後場で分ける機能を検討する余地はアリ
        :param ts_start:
        :param ts_end:
        :return:
        """
        self.chart.setXRange(ts_start, ts_end)
        # self.chart2.setXRange(ts_start, ts_end)

    def setTitle(self, title: str):
        """
        チャートのタイトルを設定
        :param title:
        :return:
        """
        self.chart.setTitle(title)
