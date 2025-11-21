import pandas as pd
from matplotlib import (
    dates as mdates,
    font_manager as fm,
    pyplot as plt,
)
from matplotlib.backends.backend_qtagg import (
    NavigationToolbar2QT as NavigationToolbar,
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.figure import Figure

from structs.res import AppRes


class Chart(FigureCanvas):
    """
    チャート用 FigureCanvas の雛形
    """

    def __init__(self, res: AppRes):
        # フォント設定
        fm.fontManager.addfont(res.path_monospace)
        font_prop = fm.FontProperties(fname=res.path_monospace)
        plt.rcParams["font.family"] = font_prop.get_name()
        plt.rcParams["font.size"] = 11
        # ダークモードの設定
        # plt.style.use("dark_background")

        # Figure オブジェクト
        self.figure = Figure()
        # 軸領域
        self.ax = self.figure.add_subplot(111)

        super().__init__(self.figure)


class MplChart(FigureCanvas):
    def __init__(self):
        self.fig = Figure()
        super().__init__(self.fig)

        # Font setting
        FONT_PATH = 'fonts/RictyDiminished-Regular.ttf'
        fm.fontManager.addfont(FONT_PATH)
        font_prop = fm.FontProperties(fname=FONT_PATH)
        plt.rcParams['font.family'] = font_prop.get_name()
        plt.rcParams['font.size'] = 14

        # dark mode
        # plt.style.use('dark_background')

        # Plot margin
        self.fig.subplots_adjust(
            left=0.075,
            right=0.98,
            top=0.95,
            bottom=0.05,
        )

        # Axes
        self.ax = dict()

    def clearAxes(self):
        axs = self.fig.axes
        for ax in axs:
            ax.cla()
            ax.grid()

    def initAxes(self, ax, n: int):
        if n > 1:
            gs = self.fig.add_gridspec(
                n, 1,
                wspace=0.0, hspace=0.0,
                height_ratios=[3 if i == 0 else 1 for i in range(n)]
            )
            for i, axis in enumerate(gs.subplots(sharex='col')):
                ax[i] = axis
                ax[i].grid()
        else:
            ax[0] = self.fig.add_subplot()
            ax[0].grid()

    def initChart(self, n: int):
        self.removeAxes()
        self.initAxes(self.ax, n)

    def refreshDraw(self):
        self.fig.canvas.draw()

    def removeAxes(self):
        axs = self.fig.axes
        for ax in axs:
            ax.remove()


class ChartNavigation(NavigationToolbar):
    def __init__(self, chart: FigureCanvas):
        super().__init__(chart)


class TickChart(Chart):
    """
    ティックチャート用
    """

    def __init__(self, res: AppRes):
        super().__init__(res)
        # 余白設定
        self.figure.subplots_adjust(
            left=0.05,
            right=0.95,
            top=0.9,
            bottom=0.08,
        )
        # タイムスタンプへ時差を加算用（Asia/Tokyo)
        self.tz = 9. * 60 * 60

        self.clearPlot()

    def clearPlot(self):
        self.ax.cla()
        self.ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%H:%M")
        )
        self.ax.grid()

    def updateData(self, df: pd.DataFrame, title: str):
        # トレンドライン（株価とVWAP）
        df.index = [pd.Timestamp(ts + self.tz, unit='s') for ts in df["Time"]]
        ser_price = df["Price"]
        ser_vwap = df["VWAP"]
        ser_ma060 = df["MA060"]
        ser_ma300 = df["MA300"]

        # 消去
        self.ax.cla()

        # プロット
        self.ax.plot(ser_price, linewidth=0.5, linestyle="solid", label="Price")
        self.ax.plot(ser_vwap, linewidth=0.5, linestyle="solid", label="VWAP")
        self.ax.plot(ser_ma060, linewidth=1, linestyle="dotted", label="MA060")
        self.ax.plot(ser_ma300, linewidth=1, linestyle="dotted", label="MA300")

        self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        self.ax.grid(True, lw=0.5)
        self.ax.legend(fontsize=9)
        self.ax.set_title(title)

        # 再描画
        self.draw()
