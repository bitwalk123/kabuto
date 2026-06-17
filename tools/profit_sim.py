import os

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

        self.toolbar = toolbar = ProfitSimulatorToolbar(res)
        toolbar.sendDataFrame.connect(self.on_plot)
        self.addToolBar(toolbar)

        self.trend = trend = ProfitReviewChart(res)
        trend.notifySelection.connect(self.on_selection)
        trend.initChart()
        self.setCentralWidget(trend)

        self.navtoolbar = navtoolbar = ProfitReviewChartNavigation(res, trend)
        self.addToolBar(Qt.ToolBarArea.BottomToolBarArea, navtoolbar)

        self.dock = dock = ProfitSimulatorDock(res)
        dock.requestClearSelection.connect(self.on_clear_selection)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def on_plot(self, df: pd.DataFrame, title:str):
        print(df.columns)
        self.trend.plot(df, title)
        self.dock.setDataFrame(df)

    def on_selection(self, dt1: pd.Timestamp, dt2: pd.Timestamp):
        self.dock.setTimeRange(dt1, dt2)

    def on_clear_selection(self):
        self.navtoolbar.setRectActive(False)

    """
    def on_selector_active(self, state: bool):
        if state:
            self.trend.setSelectorActive(False)
        else:
            self.trend.clearVSpan()
    """
