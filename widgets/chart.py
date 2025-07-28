import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from matplotlib.backends.backend_qtagg import (
    NavigationToolbar2QT as NavigationToolbar,
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.figure import Figure

from structs.res import AppRes


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


class TrendChart(FigureCanvas):
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
        self.figure.subplots_adjust(
            left=0.075,
            right=0.99,
            top=0.9,
            bottom=0.08,
        )
        super().__init__(self.figure)

        self.ax = self.figure.add_subplot(111)
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        self.ax.grid(True, lw=0.5)
