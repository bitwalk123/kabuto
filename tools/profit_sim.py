import os

import matplotlib as mpl
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon

from structs.res import AppRes
from tools.profit_sim_charts import ProfitReviewChart, ProfitReviewChartNavigation
from tools.profit_sim_widgets import ProfitSimulatorToolbar, ProfitSimulatorDock
from widgets.containers import MainWindow


class ProfitSimulator(MainWindow):
    def __init__(self):
        super().__init__()
        self.res = res = AppRes()
        self.setWindowIcon(QIcon(os.path.join(res.dir_image, "profit.png")))
        self.setWindowTitle("Profit Simulator")

        self.dt_start, self.dt_end = (None, None)
        self.p1, self.p2 = (None, None)
        self.toolbar = toolbar = ProfitSimulatorToolbar(res)
        toolbar.requestClearSelection.connect(self.on_clear_selection)
        toolbar.requestSelectorActive.connect(self.on_selector_active)
        toolbar.sendDataFrame.connect(self.on_plot)
        self.addToolBar(toolbar)

        self.trend = trend = ProfitReviewChart(res)
        trend.notifySelection.connect(self.on_selection)
        trend.initChart()
        self.setCentralWidget(trend)

        navtoolbar = ProfitReviewChartNavigation(res, trend)
        self.addToolBar(Qt.ToolBarArea.BottomToolBarArea, navtoolbar)

        self.dock = dock = ProfitSimulatorDock(res)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        # self.setCentralWidget(base)

    def clearVSpan(self):
        if type(self.p1) is mpl.patches.Rectangle:
            self.p1.remove()
            self.p1 = None
        if type(self.p2) is mpl.patches.Rectangle:
            self.p2.remove()
            self.p2 = None

    def on_plot(self, df: pd.DataFrame):
        print(df.columns)
        self.trend.plot(df)

    def on_clear_selection(self):
        self.trend.clearSelection()

    def on_selection(self, dt1: pd.Timestamp, dt2: pd.Timestamp):
        self.toolbar.setTimeRange(dt1, dt2)
        if self.dt_start is not None:
            self.clearVSpan()
            self.p1 = self.trend.ax[0].axvspan(self.dt_start, dt1, color="gray", alpha=0.25)
            self.p2 = self.trend.ax[0].axvspan(dt2, self.dt_end, color="gray", alpha=0.25)
            self.trend.refreshDraw()

    def on_selector_active(self, state: bool):
        self.trend.setSelectorActive(state)
        if state:
            self.clearVSpan()
            self.trend.refreshDraw()
