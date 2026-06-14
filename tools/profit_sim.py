import os

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QDockWidget,
    QMainWindow,
    QVBoxLayout,
)

from structs.res import AppRes
from tools.profit_sim_charts import ReviewChart
from tools.profit_sim_funcs import get_x_range, get_y_range
from tools.profit_sim_widgets import BaseWidget, ProfitSimulatorToolbar


class ProfitSimulator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.res = res = AppRes()
        self.setWindowIcon(QIcon(os.path.join(res.dir_image, "profit.png")))
        self.setWindowTitle("Profit Simulator")

        self.toolbar = toolbar = ProfitSimulatorToolbar(res)
        toolbar.requestClearSelection.connect(self.on_clear_selection)
        toolbar.requestSelectorActive.connect(self.on_selector_active)
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
        n = 2
        i = 0
        self.trend.ax[i].plot(df["price"], zorder=10, linewidth=0.25, color="black")
        self.trend.ax[i].plot(df["ma1"], zorder=20, linewidth=0.25, color="#080")
        self.trend.ax[i].plot(df["ma2"], zorder=30, linewidth=0.75, color="#f80")
        self.trend.ax[i].plot(df["vwap"], zorder=40, linewidth=0.5, color="#808")
        self.trend.ax[i].set_xlim(*get_x_range(df))
        self.trend.ax[i].set_ylim(*get_y_range(df["price"]))
        self.trend.ax[i].set_ylabel("株    価")

        i += 1
        self.trend.ax[i].plot(df["profit_max"], linewidth=0.75, color="red")
        self.trend.ax[i].plot(df["profit"], linewidth=0.25, color="magenta")
        self.trend.ax[i].set_ylabel("含み損益")

        for i in range(n):
            list_ma_gc = df[0 < df["ma_gc"]].index
            list_ma_dc = df[0 < df["ma_dc"]].index
            for i in range(n):
                # ゴールデン・クロス
                for t in list_ma_gc:
                    # cname = "#f00" if 0 < df.at[t, "ma_gc"] else "#00f"
                    self.trend.ax[i].axvline(x=t, c="#f00", ls="solid", alpha=0.25, lw=0.75)

                # デッド・クロス
                for t in list_ma_dc:
                    # cname = "#00f" if 0 < df.at[t, "ma_dc"] else "#f00"
                    self.trend.ax[i].axvline(x=t, c="#00f", ls="solid", alpha=0.25, lw=0.75)

        # プロットを更新
        self.trend.refreshDraw()

    def on_clear_selection(self):
        self.trend.clearSelection()

    def on_selection(self, dt1: pd.Timestamp, dt2: pd.Timestamp):
        self.toolbar.setTimeRange(dt1, dt2)

    def on_selector_active(self, state: bool):
        self.trend.setSelectorActive(state)
