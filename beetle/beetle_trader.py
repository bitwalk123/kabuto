import logging

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow

from modules.chart import TrendChart
from structs.res import AppRes


class Trader(QMainWindow):
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
        # データ点用のカウンター
        self.count_data = 0

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
        # self.dock = dock = DockTrader(res, code)
        # self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

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

    def setPlotData(self, ts: float, price: float):
        """
        PSAR に関連するデータをプロット
        :param ts:
        :param price:
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
        self.latest_point.set_ydata([price])

        # ---------------------------------------------------------------------
        # 現在価格（スムージングした線に変更予定）
        # ---------------------------------------------------------------------
        self.x_data[self.count_data] = x
        self.y_data[self.count_data] = price
        self.count_data += 1
        self.trend_line.set_xdata(self.x_data[0:self.count_data])
        self.trend_line.set_ydata(self.y_data[0:self.count_data])

        # 再描画
        self.chart.reDraw()

        # ---------------------------------------------------------------------
        # トレンド情報をドックに設定
        # ---------------------------------------------------------------------
        # self.dock.setTrend(ret)

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
