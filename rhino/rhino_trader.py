import logging

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow

from rhino.rhino_chart import TrendChart
from rhino.rhino_dock import DockRhinoTrader
from rhino.rhino_psar import PSARObject
from structs.res import AppRes


class RhinoTrader(QMainWindow):
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
        self.ys_data = np.empty(self.max_data_points, dtype=np.float64)
        # データ点用のカウンター
        self.count_data = 0

        # bull（上昇トレンド）
        self.x_bull = np.empty(self.max_data_points, dtype=pd.Timestamp)
        self.y_bull = np.empty(self.max_data_points, dtype=np.float64)
        # bull 用のカウンター
        self.count_bull = 0

        # bear（下降トレンド）
        self.x_bear = np.empty(self.max_data_points, dtype=pd.Timestamp)
        self.y_bear = np.empty(self.max_data_points, dtype=np.float64)
        # bear 用のカウンター
        self.count_bear = 0

        #
        #######################################################################

        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        #  UI
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # ウィンドウのサイズ制約
        self.setMinimumWidth(1200)
        self.setFixedHeight(300)

        # ---------------------------------------------------------------------
        # 右側のドック
        # ---------------------------------------------------------------------
        self.dock = dock = DockRhinoTrader(res, code)
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
            linewidth=1
        )

        # bull（Parabolic SAR 上昇トレンド）
        self.trend_bull, = self.chart.ax.plot(
            [], [],
            marker='o',
            markersize=1,
            linewidth=0,
            color='magenta'
        )

        # bear（Parabolic SAR 下降トレンド）
        self.trend_bear, = self.chart.ax.plot(
            [], [],
            marker='o',
            markersize=1,
            linewidth=0,
            color='cyan'
        )

    def getTimePrice(self) -> pd.DataFrame:
        """
        保持している時刻、株価情報をデータフレームで返す。
        :return:
        """
        # タイムスタンプ の Time 列は self.tz を考慮
        return pd.DataFrame({
            "Time": [t.timestamp() - self.tz for t in self.x_data[0: self.count_data]],
            "Price": self.y_data[0: self.count_data]
        })

    def setLastCloseLine(self, price_close: float):
        """
        前日終値ラインの描画
        :param price_close:
        :return:
        """
        self.chart.ax.axhline(y=price_close, color="red", linewidth=0.75)

    def setPlotData(self, ts: float, ret: PSARObject):
        """
        PSAR に関連するデータをプロット
        :param ts:
        :param ret:
        :return:
        """

        # ---------------------------------------------------------------------
        # ts（タイムスタンプ）から、Matplotlib 用の値＝タイムスタンプ（時差込み）に変換
        # ---------------------------------------------------------------------
        x = pd.Timestamp(ts + self.tz, unit='s')

        # ---------------------------------------------------------------------
        # 最新の株価
        # ---------------------------------------------------------------------
        self.latest_point.set_xdata([x])
        self.latest_point.set_ydata([ret.price])

        # ---------------------------------------------------------------------
        # 現在価格（スムージングした線に変更予定）
        # ---------------------------------------------------------------------
        self.x_data[self.count_data] = x
        self.y_data[self.count_data] = ret.price
        self.ys_data[self.count_data] = ret.ys
        self.count_data += 1
        self.trend_line.set_xdata(self.x_data[0:self.count_data])
        self.trend_line.set_ydata(self.ys_data[0:self.count_data])

        # ---------------------------------------------------------------------
        # Parabolic SAR のトレンド点
        # ---------------------------------------------------------------------
        if 0 < ret.trend:
            self.x_bull[self.count_bull] = x
            self.y_bull[self.count_bull] = ret.psar
            self.count_bull += 1
            self.trend_bull.set_xdata(self.x_bull[0:self.count_data])
            self.trend_bull.set_ydata(self.y_bull[0:self.count_data])
        elif ret.trend < 0:
            self.x_bear[self.count_bear] = x
            self.y_bear[self.count_bear] = ret.psar
            self.count_bear += 1
            self.trend_bear.set_xdata(self.x_bear[0:self.count_data])
            self.trend_bear.set_ydata(self.y_bear[0:self.count_data])
        else:
            # ret.trend == 0 の時
            pass

        # 再描画
        self.chart.reDraw()

        # ---------------------------------------------------------------------
        # トレンド情報をドックに設定
        # ---------------------------------------------------------------------
        self.dock.setTrend(ret)

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
