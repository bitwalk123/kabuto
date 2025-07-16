import datetime
import logging

import numpy as np
import pandas as pd
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib import dates as mdates
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow

from beetle.beetle_psar import PSARObject
from structs.res import AppRes
from widgets.docks import DockTrader


class Trader(QMainWindow):
    def __init__(self, res: AppRes, ticker: str):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.res = res
        self.ticker = ticker

        # タイムスタンプへ時差を加算・減算用（Asia/Tokyo)
        self.tz = 9. * 60 * 60

        self.setFixedSize(1200, 300)
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
        self.counter_data = 0

        # bull（上昇トレンド）
        self.x_bull = np.empty(self.max_data_points, dtype=pd.Timestamp)
        self.y_bull = np.empty(self.max_data_points, dtype=np.float64)
        # bull 用のカウンター
        self.counter_bull = 0

        # bear（下降トレンド）
        self.x_bear = np.empty(self.max_data_points, dtype=pd.Timestamp)
        self.y_bear = np.empty(self.max_data_points, dtype=np.float64)
        # bear 用のカウンター
        self.counter_bear = 0

        #
        #######################################################################

        # ---------------------------------------------------------------------
        # 右側のドック
        # ---------------------------------------------------------------------
        self.dock = dock = DockTrader(res, ticker)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

        # Matplotlib 用設定
        FONT_PATH = "fonts/RictyDiminished-Regular.ttf"
        fm.fontManager.addfont(FONT_PATH)

        # FontPropertiesオブジェクト生成（名前の取得のため）
        font_prop = fm.FontProperties(fname=FONT_PATH)
        font_prop.get_name()

        # フォント設定
        plt.rcParams["font.family"] = font_prop.get_name()
        plt.rcParams["font.size"] = 12

        # ダークモードの設定
        plt.style.use("dark_background")

        self.figure = Figure()

        # Figure オブジェクトの余白設定
        self.figure.subplots_adjust(
            left=0.075,
            right=0.99,
            top=0.9,
            bottom=0.08,
        )

        # ---------------------------------------------------------------------
        # チャートインスタンス (FigureCanvas)
        # ---------------------------------------------------------------------
        self.chart = chart = FigureCanvas(self.figure)
        self.setCentralWidget(chart)

        # 描画用インスタンス (ax）
        self.ax = self.figure.add_subplot(111)
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        self.ax.grid(True)

        # トレンドライン（株価）
        self.trend_line, = self.ax.plot(
            [], [],
            color='lightgray',
            linewidth=0.5
        )

        # bull（Parabolic SAR 上昇トレンド）
        self.trend_bull, = self.ax.plot(
            [], [],
            marker='o',
            markersize=2,
            linewidth=0,
            color='magenta'
        )

        # bear（Parabolic SAR 下降トレンド）
        self.trend_bear, = self.ax.plot(
            [], [],
            marker='o',
            markersize=2,
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
            "Time": [x - self.tz for x in self.x_data[0: self.counter_data]],
            "Price": self.y_data[0: self.counter_data]
        })

    def setLastCloseLine(self, price_close: float):
        """
        前日終値ラインの描画
        :param price_close:
        :return:
        """
        self.ax.axhline(y=price_close, color="red", linewidth=0.75)

    def setPSAR(self, ts: float, ret: PSARObject):
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
        # 現在価格（スムージングした線に変更予定）
        # ---------------------------------------------------------------------
        self.x_data[self.counter_data] = x
        self.y_data[self.counter_data] = ret.price
        self.counter_data += 1
        self.trend_line.set_xdata(self.x_data[0:self.counter_data])
        self.trend_line.set_ydata(self.y_data[0:self.counter_data])

        # ---------------------------------------------------------------------
        # Parabolic SAR のトレンド点
        # ---------------------------------------------------------------------
        if 0 < ret.trend:
            self.x_bull[self.counter_bull] = x
            self.y_bull[self.counter_bull] = ret.psar
            self.counter_bull += 1
            self.trend_bull.set_xdata(self.x_bull[0:self.counter_data])
            self.trend_bull.set_ydata(self.y_bull[0:self.counter_data])
        elif ret.trend < 0:
            self.x_bear[self.counter_bear] = x
            self.y_bear[self.counter_bear] = ret.psar
            self.counter_bear += 1
            self.trend_bear.set_xdata(self.x_bear[0:self.counter_data])
            self.trend_bear.set_ydata(self.y_bear[0:self.counter_data])
        else:
            # ret.trend == 0 の時
            pass

        # Recompute the data limits based on current artists.
        self.ax.relim()
        # y軸のみオートスケール
        self.ax.autoscale_view(scalex=False, scaley=True)  # X軸は固定、Y軸は自動
        # Canvas を再描画
        self.chart.draw()

        # ---------------------------------------------------------------------
        # トレンド情報をドックに設定
        # ---------------------------------------------------------------------
        self.dock.setTrend(ret.trend, ret.epupd)

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
        self.ax.set_xlim(dt_start, dt_end)

    def setChartTitle(self, title: str):
        """
        チャートのタイトルを設定
        :param title:
        :return:
        """
        # self.chart.setTitle(title)
        self.ax.set_title(title)
