import matplotlib as mpl
import pandas as pd
from PySide6.QtCore import Signal
from matplotlib import font_manager as fm, dates as mdates
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector

from tools.profit_sim_funcs import to_pd_dt


class ReviewChart(FigureCanvas):
    notifySelection = Signal(pd.Timestamp, pd.Timestamp)

    def __init__(self):
        self.fig = Figure()
        super().__init__(self.fig)
        # Font setting
        FONT_PATH = 'fonts/RictyDiminished-Regular.ttf'
        fm.fontManager.addfont(FONT_PATH)
        font_prop = fm.FontProperties(fname=FONT_PATH)
        # ★ 全体フォント適用
        mpl.rcParams['font.family'] = font_prop.get_name()

        # Plot margin
        self.fig.subplots_adjust(
            left=0.05,
            right=0.99,
            top=0.9,
            bottom=0.075,
        )

        # Axes
        self.ax = dict()

        # Selector
        self.selector = None

    def clearAxes(self):
        axs = self.fig.axes
        for ax in axs:
            ax.cla()
            ax.grid(axis="y")

    def clearSelection(self):
        self.selector.clear()

    def initAxes(self):
        gs = self.fig.add_gridspec(
            2, 1,
            wspace=0.0, hspace=0.0,
            height_ratios=[2 if i == 0 else 1 for i in range(2)]
        )
        for i, axis in enumerate(gs.subplots(sharex='col')):
            self.ax[i] = axis
            self.ax[i].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            self.ax[i].grid(axis="y")

        # Selector
        self.selector = RectangleSelector(
            self.ax[0], self.selection_callback, useblit=True,
            button=[1],  # disable middle & right buttons
            minspanx=4, minspany=4, spancoords='pixels', interactive=True,
            props=dict(facecolor='pink', edgecolor='red', alpha=0.1, fill=True)
        )

    def initChart(self):
        self.removeAxes()
        self.initAxes()

    def refreshDraw(self):
        self.fig.canvas.draw_idle()

    def removeAxes(self):
        for ax in list(self.fig.axes):
            ax.remove()

    def selection_callback(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        dt1 = to_pd_dt(x1)
        dt2 = to_pd_dt(x2)

        # print(f"({dt1}, {y1: 3.2f}) --> ({dt2}, {y2: 3.2f})")
        self.notifySelection.emit(dt1, dt2)

    def setSelectorActive(self, state: bool):
        self.selector.set_active(state)
