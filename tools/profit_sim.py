import os

import matplotlib as mpl
import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QDockWidget,
    QMainWindow,
    QVBoxLayout,
)
from matplotlib import (
    dates as mdates,
    font_manager as fm,
)
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector

from structs.res import AppRes
from tools.profit_sim_funcs import get_x_range, get_y_range, to_pd_dt
from tools.profit_sim_widgets import BaseWidget, ProfitSimulatorToolbar


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
            ax.grid()

    def initAxes(self):
        gs = self.fig.add_gridspec(
            2, 1,
            wspace=0.0, hspace=0.0,
            height_ratios=[2 if i == 0 else 1 for i in range(2)]
        )
        for i, axis in enumerate(gs.subplots(sharex='col')):
            self.ax[i] = axis
            self.ax[i].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            self.ax[i].grid()

        # Selector
        self.selector = RectangleSelector(
            self.ax[0], self.selection, useblit=True,
            button=[1],  # disable middle & right buttons
            minspanx=5, minspany=5, spancoords='pixels', interactive=True,
            props=dict(facecolor='pink', edgecolor='red', alpha=0.2, fill=True)
        )

    def initChart(self):
        self.removeAxes()
        self.initAxes()

    def refreshDraw(self):
        self.fig.canvas.draw()

    def removeAxes(self):
        for ax in list(self.fig.axes):
            ax.remove()

    def selection(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        dt1 = to_pd_dt(x1)
        dt2 = to_pd_dt(x2)

        # print(f"({dt1}, {y1: 3.2f}) --> ({dt2}, {y2: 3.2f})")
        self.notifySelection.emit(dt1, dt2)

    def resetSelection(self):
        if self.selector:
            self.selector.set_active(False)
            self.selector.update()
            self.fig.canvas.draw_idle()
            self.selector.set_active(True)


class ProfitSimulator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.res = res = AppRes()
        self.setWindowIcon(QIcon(os.path.join(res.dir_image, "profit.png")))
        self.setWindowTitle("Profit Simulator")

        toolbar = ProfitSimulatorToolbar(res)
        toolbar.sendDataFrame.connect(self.on_plot)

        base = BaseWidget(res)
        layout = QVBoxLayout()
        base.setLayout(layout)
        self.trend = trend = ReviewChart()
        trend.notifySelection.connect(self.on_selection)
        trend.initChart()
        layout.addWidget(trend)

        dock = QDockWidget("Controller")

        self.addToolBar(toolbar)
        self.setCentralWidget(base)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def on_plot(self, df: pd.DataFrame):
        print(df.columns)

        self.trend.ax[0].plot(df["price"], linewidth=0.25, color="black")
        # self.trend.ax[0].relim()
        # self.trend.ax[0].autoscale_view(scalex=False, scaley=True)
        self.trend.ax[0].set_xlim(*get_x_range(df))
        self.trend.ax[0].set_ylim(*get_y_range(df["price"]))

        self.trend.ax[1].plot(df["profit_max"], linewidth=0.75, color="red")
        self.trend.ax[1].plot(df["profit"], linewidth=0.25, color="magenta")
        self.trend.refreshDraw()

    def on_selection(self, dt1: pd.Timestamp, dt2: pd.Timestamp):
        print(dt1, dt2)
