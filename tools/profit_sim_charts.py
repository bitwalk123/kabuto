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
        self.setMinimumWidth(res.profit_width)
        self.setFixedHeight(res.profit_height)

        # プロットの取引時間、寄り付きから大引けまで
        self.dt_start, self.dt_end = (None, None)
        # 選択範囲
        self.p1, self.p2 = (None, None)

        # Font setting
        FONT_PATH = 'fonts/RictyDiminished-Regular.ttf'
        fm.fontManager.addfont(FONT_PATH)
        font_prop = fm.FontProperties(fname=FONT_PATH)
        # ★ 全体フォント適用
        mpl.rcParams['font.family'] = font_prop.get_name()
        mpl.rcParams["font.size"] = 8

        # Plot margin
        self.fig.subplots_adjust(
            left=0.1,
            right=0.99,
            top=0.94,
            bottom=0.06,
        )

        # rows of plot
        self.rows = 3

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
            self.ax[i].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
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

    def plot(self, df: pd.DataFrame, title: str, path_csv: str):
        # 保存先を設定
        self.set_save_config(path_csv)

        # --- 株価、VWAP線、移動平均線 ---
        i = 0
        # チャートタイトル
        self.ax[i].set_title(title)

        self.ax[i].plot(df["price"], zorder=10, linewidth=0.25, color="black")
        self.ax[i].plot(df["ma1"], zorder=20, linewidth=1, color="#080")
        self.ax[i].plot(df["ma2"], zorder=30, linewidth=1, color="#f80")
        self.ax[i].plot(df["vwap"], zorder=40, linewidth=0.5, color="#808")
        # プロットの取引時間、寄り付きから大引けまで
        self.dt_start, self.dt_end = get_x_range(df)
        self.ax[i].set_xlim(self.dt_start, self.dt_end)
        # y軸のスケーリング
        self.ax[i].set_ylim(*get_y_range(df["price"]))
        # y軸ラベル (1)
        self.ax[i].set_ylabel("株    価")

        # --- モメンタム ---
        i += 1
        self.ax[i].plot(df["momentum"], zorder=20, linewidth=0.75, color="#840")
        self.ax[i].axhline(y=0, c="#000", ls="solid", alpha=0.5, lw=0.75)
        # y軸ラベル (2)
        self.ax[i].set_ylabel("モメンタム")

        # --- 含み損益 ---
        i += 1
        x = df.index
        y1 = df["profit"]
        y2 = df["profit_max"]
        self.ax[i].fill_between(x, 0, y1, where=(0 < y1), fc="#fbb", ec="#f00", alpha=0.5, lw=0.5, zorder=10)
        self.ax[i].fill_between(x, 0, y1, where=(y1 < 0), fc="#bbf", ec="#00f", alpha=0.5, lw=0.5, zorder=10)
        self.ax[i].plot(y2, linewidth=0.75, color="#800", zorder=60)
        # y軸ラベル (3)
        self.ax[i].set_ylabel("含み損益")

        # --- クロス・シグナル ---
        list_ma_gc = df[0 < df["ma_gc"]].index
        list_ma_dc = df[0 < df["ma_dc"]].index
        for i in range(self.rows):
            # ゴールデン・クロス
            for t in list_ma_gc:
                self.ax[i].axvline(x=t, zorder=100, c="#f00", ls="solid", alpha=0.5, lw=0.5)
            # デッド・クロス
            for t in list_ma_dc:
                self.ax[i].axvline(x=t, zorder=100, c="#00f", ls="solid", alpha=0.5, lw=0.5)

        # --- プロットを更新 ---
        self.refreshDraw()

    def set_save_config(self, path_csv: str):
        # ディレクトリはデータファイルと同じ
        mpl.rcParams["savefig.directory"] = os.path.dirname(path_csv)
        # 保存ファイルは、拡張子以外はデータファイルと同じ
        basename_without_ext = os.path.splitext(os.path.basename(path_csv))[0]
        self.get_default_filename = lambda: basename_without_ext

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
        self.setVSpan(dt1, dt2)
        self.notifySelection.emit(dt1, dt2)

    def setSelectorActive(self, state: bool):
        self.selector.set_active(state)

    def setVSpan(self, dt1: pd.Timestamp, dt2: pd.Timestamp):
        if self.dt_start is not None:
            self.clearVSpan()
            self.p1 = self.ax[0].axvspan(self.dt_start, dt1, color="gray", alpha=0.25)
            self.p2 = self.ax[0].axvspan(dt2, self.dt_end, color="gray", alpha=0.25)
            self.refreshDraw()

    def clearVSpan(self):
        if type(self.p1) is mpl.patches.Rectangle:
            self.p1.remove()
            self.p1 = None
        if type(self.p2) is mpl.patches.Rectangle:
            self.p2.remove()
            self.p2 = None
        self.refreshDraw()


class ProfitReviewChartNavigation(NavigationToolbar):
    def __init__(self, res: AppRes, trend: ProfitReviewChart):
        super().__init__(trend)
        # print(dir(self))
        self.trend: ProfitReviewChart = trend

        icon = QIcon(os.path.join(res.dir_image, "rect.png"))
        self.action_rect = action_rect = QAction(icon, "Rectangle", self)
        action_rect.setCheckable(True)
        action_rect.toggled.connect(self.setRectActive)

        # self.addAction(review_action)
        actions = self.actions()
        n = len(actions)
        self.insertAction(actions[n - 1], action_rect)

    def setRectActive(self, state: bool):
        # Zoomモードなら解除
        if self._actions["zoom"].isChecked():
            self._actions["zoom"].trigger()
        # Panモードなら解除
        if self._actions["pan"].isChecked():
            self._actions["pan"].trigger()

        # print(f"User action toggled: {state}")
        self.trend.selector.set_active(state)
        if not state:
            self.trend.clearSelection()
