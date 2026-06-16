import os

import matplotlib as mpl
import pandas as pd
from PySide6.QtCore import Signal, QMargins
from PySide6.QtGui import QAction, QIcon
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
from matplotlib.widgets import RectangleSelector

from structs.res import AppRes
from tools.profit_sim_funcs import (
    get_x_range,
    get_y_range,
    to_pd_dt,
)


class ProfitReviewChart(FigureCanvas):
    notifySelection = Signal(pd.Timestamp, pd.Timestamp)

    def __init__(self, res: AppRes):
        self.fig = Figure()
        super().__init__(self.fig)
        self.setContentsMargins(QMargins(0, 0, 0, 0))
        self.setFixedHeight(res.profit_height)
        self.setMinimumWidth(res.profit_width)

        # Font setting
        FONT_PATH = 'fonts/RictyDiminished-Regular.ttf'
        fm.fontManager.addfont(FONT_PATH)
        font_prop = fm.FontProperties(fname=FONT_PATH)
        # ★ 全体フォント適用
        mpl.rcParams['font.family'] = font_prop.get_name()
        mpl.rcParams["font.size"] = 9

        # Plot margin
        self.fig.subplots_adjust(
            left=0.1,
            right=0.99,
            top=0.9,
            bottom=0.075,
        )

        # rows of plot
        self.rows = 2

        # Axes
        self.ax = dict()

        # Selector
        self.selector: RectangleSelector | None = None

    def clearAxes(self):
        axs = self.fig.axes
        for ax in axs:
            ax.cla()
            ax.grid(axis="y")

    def clearSelection(self):
        self.selector.clear()

    def initAxes(self):
        gs = self.fig.add_gridspec(
            self.rows, 1,
            wspace=0.0, hspace=0.0,
            height_ratios=[2 if i == 0 else 1 for i in range(self.rows)]
        )
        for i, axis in enumerate(gs.subplots(sharex='col')):
            self.ax[i] = axis
            self.ax[i].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            self.ax[i].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:>6,.0f}"))
            self.ax[i].grid(axis="y")

        # Selector
        self.selector = RectangleSelector(
            self.ax[0], self.selection_callback, useblit=True,
            button=[1],  # disable middle & right buttons
            minspanx=4, minspany=4, spancoords='pixels', interactive=True,
            props=dict(facecolor='pink', edgecolor='red', alpha=0.1, fill=True)
        )
        self.selector.set_active(False)

    def initChart(self):
        self.removeAxes()
        self.initAxes()

    def plot(self, df: pd.DataFrame):
        i = 0
        self.ax[i].plot(df["price"], zorder=10, linewidth=0.25, color="black")
        self.ax[i].plot(df["ma1"], zorder=20, linewidth=0.25, color="#080")
        self.ax[i].plot(df["ma2"], zorder=30, linewidth=0.75, color="#f80")
        self.ax[i].plot(df["vwap"], zorder=40, linewidth=0.5, color="#808")
        self.dt_start, self.dt_end = get_x_range(df)
        self.ax[i].set_xlim(self.dt_start, self.dt_end)
        self.ax[i].set_ylim(*get_y_range(df["price"]))
        self.ax[i].set_ylabel("株    価")

        i += 1
        self.ax[i].plot(df["profit_max"], linewidth=0.75, color="red")
        self.ax[i].plot(df["profit"], linewidth=0.25, color="magenta")
        self.ax[i].set_ylabel("含み損益")

        list_ma_gc = df[0 < df["ma_gc"]].index
        list_ma_dc = df[0 < df["ma_dc"]].index
        for i in range(self.rows):
            # ゴールデン・クロス
            for t in list_ma_gc:
                self.ax[i].axvline(x=t, c="#f00", ls="solid", alpha=0.25, lw=0.75)
            # デッド・クロス
            for t in list_ma_dc:
                self.ax[i].axvline(x=t, c="#00f", ls="solid", alpha=0.25, lw=0.75)

        # プロットを更新
        self.refreshDraw()

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


class ProfitReviewChartNavigation(NavigationToolbar):
    def __init__(self, res: AppRes, canvas: FigureCanvas):
        super().__init__(canvas)
        # print(dir(self))
        self.canvas = canvas

        icon = QIcon(os.path.join(res.dir_image, "rect.png"))
        self.action_user = action_user = QAction(icon, "User", self)
        action_user.setCheckable(True)
        action_user.toggled.connect(self.on_action_user)

        # self.addAction(review_action)
        actions = self.actions()
        n = len(actions)
        self.insertAction(actions[n - 1], action_user)

    def on_action_user(self, state: bool):
        # Zoomモードなら解除
        if self._actions["zoom"].isChecked():
            self._actions["zoom"].trigger()
        # Panモードなら解除
        if self._actions["pan"].isChecked():
            self._actions["pan"].trigger()

        print(f"User action toggled: {state}")
        self.canvas.selector.set_active(state)
