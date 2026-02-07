import logging

import pandas as pd

from structs.res import AppRes
from widgets.mpl_charts import ChartNavigation, TickChart
from widgets.containers import MainWindow
from widgets.statusbars import StatusBar


class WinTick(MainWindow):
    def __init__(self, res: AppRes):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.res = res
        self.setFixedWidth(1800)

        # ---------------------------------------------------------------------
        # チャート
        # ---------------------------------------------------------------------
        self.chart = chart = TickChart(res)
        self.setCentralWidget(chart)

        # ---------------------------------------------------------------------
        # ステータスバー
        # ---------------------------------------------------------------------
        self.statusbar = statusbar = StatusBar(res)
        statusbar.setSizeGripEnabled(False)
        navbar = ChartNavigation(chart.canvas)
        statusbar.addWidget(navbar)
        self.setStatusBar(statusbar)

    def draw(self, df: pd.DataFrame, dict_param: dict, title: str = ""):
        self.chart.updateData(df, dict_param, title)
