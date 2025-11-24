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

from funcs.technical import calc_ma, calc_msd
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
        self.ax2 = self.ax.twinx()
        # 余白設定
        self.figure.subplots_adjust(
            left=0.05,
            right=0.95,
            top=0.9,
            bottom=0.08,
        )
        self.space = "          "
        # タイムスタンプへ時差を加算用（Asia/Tokyo)
        self.tz = 9. * 60 * 60

        self.clearPlot()

    def clearPlot(self):
        self.ax.cla()
        self.ax2.cla()
        self.ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%H:%M")
        )
        # self.ax.grid()

    def updateData(self, df: pd.DataFrame, dict_param: dict, title: str):
        # トレンドライン（株価とVWAP）
        df.index = [pd.Timestamp(ts + self.tz, unit='s') for ts in df["Time"]]

        period_mad_1 = dict_param["period_mad_1"]
        period_mad_2 = dict_param["period_mad_2"]
        period_msd = dict_param["period_msd"]
        colname_ma_1, colname_ma_2 = calc_ma(df, period_mad_1, period_mad_2)
        colname_msd = calc_msd(df, period_msd)
        ser_price = df["Price"]
        ser_ma_1 = df[colname_ma_1]
        ser_ma_2 = df[colname_ma_2]
        ser_msd = df[colname_msd]
        # print(ser_msd.describe())

        # 消去
        self.ax.cla()
        self.ax2.cla()

        # プロット　(y)
        self.ax.plot(ser_price, color="lightgray", linewidth=0.5, linestyle="solid", label="Price")
        self.ax.plot(ser_ma_1, linewidth=1, linestyle="solid", label=colname_ma_1)
        self.ax.plot(ser_ma_2, linewidth=1, linestyle="solid", label=colname_ma_2)

        self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        self.ax.grid(True, lw=0.5)

        y_min, y_max = self.ax.get_ylim()
        y_min_2 = 2 * y_min - y_max
        if y_min_2 < 0:
            y_min_2 = 0
        self.ax.set_ylim(y_min_2, y_max)
        yticks = self.ax.get_yticklabels()
        idx_min = len(yticks) // 2 - 1
        for i in range(idx_min):
            yticks[i].set_visible(False)

        self.ax.legend(fontsize=9)
        self.ax.set_ylabel(f"{self.space}Price [JPY]")
        self.ax.set_title(title)

        # プロット　(y2)
        self.ax2.plot(ser_msd, color="C2", linewidth=0.75, linestyle="solid", label=colname_msd)
        y_min, y_max = self.ax2.get_ylim()
        y_min = 0.0
        y_max_2 = 2 * y_max - y_min
        self.ax2.set_ylim(y_min, y_max_2)
        y2ticks = self.ax2.get_yticklabels()
        idx_max = len(y2ticks) // 2 + 2
        for i in range(idx_max, len(y2ticks)):
            y2ticks[i].set_visible(False)

        y_min, _ = self.ax.get_ylim()
        y2_min, _ = self.ax2.get_ylim()
        self.ax2.yaxis.set_label_position("right")
        self.ax2.set_ylabel(f"Moving σ{self.space}")

        # 再描画
        self.draw()


class TrendChart(FigureCanvas):
    """
    リアルタイム用トレンドチャート
    """

    def __init__(self, res: AppRes):
        # フォント設定
        fm.fontManager.addfont(res.path_monospace)
        font_prop = fm.FontProperties(fname=res.path_monospace)
        plt.rcParams["font.family"] = font_prop.get_name()
        plt.rcParams["font.size"] = 12
        # ダークモードの設定
        plt.style.use("dark_background")

        # Figure オブジェクト
        self.figure = Figure()
        # 余白設定
        self.figure.subplots_adjust(left=0.075, right=0.99, top=0.9, bottom=0.08)
        super().__init__(self.figure)

        self.ax = self.figure.add_subplot(111)
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        self.ax.grid(True, lw=0.5)

    def reDraw(self):
        # データ範囲を再計算
        self.ax.relim()
        # y軸のみオートスケール
        self.ax.autoscale_view(scalex=False, scaley=True)  # X軸は固定、Y軸は自動
        # 再描画
        self.draw()

    def setTitle(self, title: str):
        self.ax.set_title(title)
