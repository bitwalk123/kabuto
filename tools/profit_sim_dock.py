import pandas as pd
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QDockWidget

from structs.res import AppRes
from tools.profit_sim_widgets import TimeRange
from widgets.combos import ComboBox
from widgets.containers import Widget
from widgets.layouts import VBoxLayout


class ProfitSimulatorDock(QDockWidget):
    requestSelectedData = Signal(pd.Timestamp, pd.Timestamp)

    def __init__(self, res: AppRes):
        super().__init__()
        self.res = res

        base = Widget()
        self.setWidget(base)
        layout = VBoxLayout()
        base.setLayout(layout)

        self.trange = trange = TimeRange(res)
        trange.notifyTimeRangeFixed.connect(self.on_timerange_fixed)
        layout.addWidget(trange)

        self.combo = combo = ComboBox()
        layout.addWidget(combo)

    def clearTimeRange(self):
        self.trange.clearTimeRange()

    def on_timerange_fixed(self, dt1: pd.Timestamp, dt2: pd.Timestamp):
        self.requestSelectedData.emit(dt1, dt2)

    def setDataFrameSelected(self, df: pd.DataFrame):
        # self.df = df
        print(df)

    def setTimeRange(self, dt1: pd.Timestamp, dt2: pd.Timestamp):
        self.trange.setTimeRange(dt1, dt2)
