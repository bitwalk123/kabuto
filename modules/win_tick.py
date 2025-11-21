import logging

import pandas as pd

from funcs.technical import calc_vwap, calc_ma
from structs.res import AppRes
from widgets.charts import ChartNavigation, TickChart
from widgets.containers import MainWindow
from widgets.statusbars import StatusBar


class WinTick(MainWindow):
    def __init__(self, res: AppRes):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.res = res

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
        navbar = ChartNavigation(chart)
        statusbar.addWidget(navbar)
        self.setStatusBar(statusbar)

    def draw(self, df: pd.DataFrame, title: str = ""):
        # VWAP, MA を算出してからプロット
        calc_vwap(df)
        calc_ma(df)
        self.chart.updateData(df, title)
