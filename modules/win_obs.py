import logging

import pandas as pd

from structs.res import AppRes
from widgets.charts import ChartNavigation, ObsChart
from widgets.containers import MainWindow
from widgets.statusbars import StatusBar


class WinObs(MainWindow):
    def __init__(self, res: AppRes):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.res = res

        # ---------------------------------------------------------------------
        # チャート
        # ---------------------------------------------------------------------
        self.chart = chart = ObsChart(res)
        self.setCentralWidget(chart)

        # ---------------------------------------------------------------------
        # ステータスバー
        # ---------------------------------------------------------------------
        self.statusbar = statusbar = StatusBar(res)
        statusbar.setSizeGripEnabled(False)
        navbar = ChartNavigation(chart.canvas)
        statusbar.addWidget(navbar)
        self.setStatusBar(statusbar)

    def draw(self, df: pd.DataFrame, title: str = ""):
        self.chart.updateData(df, title)
