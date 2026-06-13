import os

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QDockWidget,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.figure import Figure

from structs.res import AppRes
from tools.profit_sim_toolbar import ProfitSimulatorToolbar


class ReviewChart(FigureCanvas):
    def __init__(self):
        self.fig = Figure()
        super().__init__(self.fig)


class ProfitSimulator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.res = res = AppRes()
        self.setWindowIcon(QIcon(os.path.join(res.dir_image, "profit.png")))
        self.setWindowTitle("Profit Simulator")

        toolbar = ProfitSimulatorToolbar(res)
        toolbar.sendDataFrame.connect(self.on_plot)
        base = QWidget()
        base.setFixedHeight(res.trend_height)
        base.setMinimumWidth(res.trend_width)
        layout = QVBoxLayout()
        base.setLayout(layout)
        trend = ReviewChart()
        layout.addWidget(trend)
        dock = QDockWidget("Controller")

        self.addToolBar(toolbar)
        self.setCentralWidget(base)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def on_plot(self, df: pd.DataFrame):
        print(df)
