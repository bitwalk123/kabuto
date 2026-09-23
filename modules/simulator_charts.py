import datetime

import matplotlib as mpl
from PySide6.QtCore import Qt, QMargins
from PySide6.QtWidgets import QMainWindow
from matplotlib import (
    dates as mdates,
    font_manager as fm,
    ticker,
)
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

from structs.res import AppRes


class SimulatorCharts(FigureCanvas):
    def __init__(self, res: AppRes):
        # 親クラスの初期化に必要な Figure インスタンスを生成
        self.fig = Figure()
        super().__init__(self.fig)
        self.res = res
        self.setMinimumWidth(1200)

        # フォント設定
        fm.fontManager.addfont(res.path_monospace)
        font_prop = fm.FontProperties(fname=res.path_monospace)
        mpl.rcParams["font.family"] = font_prop.get_name()
        mpl.rcParams["font.size"] = 11

    def plot(self, dict_info: dict):
        """
        シミュレーション結果のプロット

        :param df: プロット用のなデータ
        :param dict_info: チャートに関連する情報辞書
        """
        # self.remove_axes()  # 一旦クリア

        n = 2
        ax = dict()
        gs = self.fig.add_gridspec(
            n, 1,
            wspace=0.0, hspace=0.0,
            height_ratios=[2 if i == 0 else 1 for i in range(n)]
        )
        axes = gs.subplots(sharex="col", squeeze=False)
        for i, axis in enumerate(axes[:, 0]):
            ax[i] = axis
            ax[i].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax[i].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:>6,.0f}"))
            ax[i].xaxis.set_major_locator(mdates.HourLocator(interval=1))
            ax[i].xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[30]))

            ax[i].grid(which="major", axis="x", linestyle="-")
            ax[i].grid(which="minor", axis="x", linestyle=":")
            ax[i].grid(axis="y")
            # ax[i].grid(True)  # グリッド線の追加

        df = dict_info["technicals"]

        i = 0
        ax[i].plot(df["price"], zorder=10, linewidth=0.25, color="black")
        ax[i].plot(df["ma1"], zorder=20, linewidth=1, color="#080")
        ax[i].plot(df["ma2"], zorder=30, linewidth=1, color="#f80")
        ax[i].plot(df["vwap"], zorder=40, linewidth=0.5, color="#808")

        # チャートタイトル
        if "title" in dict_info:
            ax[i].set_title(dict_info["title"])

        td = datetime.timedelta(minutes=5)
        x_min = dict_info["mkt_start"] - td
        x_max = dict_info["mkt_end"]
        ax[i].set_xlim(x_min, x_max)

        # y軸ラベル (1)
        ax[i].set_ylabel("株    価")

        # --- 含み損益 ---
        i += 1
        x = df.index
        y1 = df["profit"]
        y2 = df["profit_max"]
        ax[i].fill_between(x, 0, y1, where=(0 < y1), fc="#fbb", ec="#f00", alpha=0.5, lw=0.5, zorder=10)
        ax[i].fill_between(x, 0, y1, where=(y1 < 0), fc="#bbf", ec="#00f", alpha=0.5, lw=0.5, zorder=10)
        ax[i].plot(y2, linewidth=0.75, color="#800", zorder=60)
        # y軸ラベル (3)
        ax[i].set_ylabel("含み損益")

        self.fig.tight_layout()  # 余白をタイトに
        self.fig.canvas.draw()  # 表示を更新

    def remove_axes(self):
        """
        Figure に追加されている Axes を削除
        """
        for ax in list(self.fig.axes):
            ax.remove()


class ChartWindow(QMainWindow):
    def __init__(self, res: AppRes):
        super().__init__()
        self.setContentsMargins(QMargins(0, 0, 0, 0))

        self.chart = chart = SimulatorCharts(res)
        self.setCentralWidget(chart)

        self.addToolBar(
            Qt.ToolBarArea.BottomToolBarArea,
            NavigationToolbar(chart)
        )

    def plot(self, dict_info: dict):
        self.chart.plot(dict_info)

    def remove_axes(self):
        self.chart.remove_axes()
