import os

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon

from structs.res import AppRes
from tools.profit_sim_charts import ProfitReviewChart, ProfitReviewChartNavigation
from tools.profit_sim_dock import ProfitSimulatorDock
from tools.profit_sim_toolbar import ProfitSimulatorToolbar
from widgets.containers import MainWindow, TabWidget


class ProfitSimulatorApp(MainWindow):
    def __init__(self):
        super().__init__()
        self.res = res = AppRes()
        self.df = pd.DataFrame()

        self.setWindowIcon(QIcon(os.path.join(res.dir_image, "profit.png")))
        self.setWindowTitle("Profit Simulator App")

        self.tick = None
        self.navtoolbar_trend = None

        self.toolbar = toolbar = ProfitSimulatorToolbar(res)
        toolbar.sendDataFrame.connect(self.on_plot)
        self.addToolBar(toolbar)

        base = TabWidget()
        base.addTab(self.gen_base_tick(), "全ティックデータ")
        self.setCentralWidget(base)

        self.dock = dock = ProfitSimulatorDock(res)
        # dock.requestSelectedData.connect(self.on_selection_fixed)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def gen_base_tick(self) -> MainWindow:
        base_tick = MainWindow()
        self.tick = tick = ProfitReviewChart(self.res)
        # tick.notifySelection.connect(self.on_selection)
        tick.initChart()
        base_tick.setCentralWidget(tick)

        self.navtoolbar_trend = toolbar = ProfitReviewChartNavigation(self.res, tick)
        base_tick.addToolBar(Qt.ToolBarArea.BottomToolBarArea, toolbar)
        return base_tick

    def on_plot(self, df: pd.DataFrame, title: str, path_csv: str):
        print(df.columns)
        self.df = df
        self.tick.plot(df, title, path_csv)
        # ドックに銘柄コードとデータフレームをセット
        code = self.toolbar.getCode()
        self.dock.setDataFrame(code, df)

    '''
    def on_selection(self, dt1: pd.Timestamp, dt2: pd.Timestamp):
        self.dock.setTimeRange(dt1, dt2)

    def on_selection_fixed(self, dt1: pd.Timestamp, dt2: pd.Timestamp):
        # チャート上の矩形領域を非アクティブに
        self.navtoolbar_trend.setRectActive(False)

        # 対象の銘柄コードを取得
        code = self.toolbar.getCode()

        # 選択された時間のデータを抽出
        df_selected = self.df[(dt1 <= self.df.index) & (self.df.index <= dt2)].copy()
        # 含み損益関連の列を 0 に
        df_selected["profit"] = 0
        df_selected["profit_max"] = 0
        df_selected["dd_ratio"] = 0
        df_selected["n_trade"] = 0
        df_selected["count_negative"] = 0

        # dock に抽出したデータを設定
        self.dock.setDataFrameSelected(code, df_selected)
    '''
