import os

import pandas as pd
from PySide6.QtCore import QMargins, Signal
from PySide6.QtGui import QIcon

from structs.res import AppRes
from widgets.buttons import Button
from widgets.containers import Widget
from widgets.labels import Label, LabelTime
from widgets.layouts import HBoxLayout


class BaseWidget(Widget):
    def __init__(self, res: AppRes):
        super().__init__()
        self.setContentsMargins(QMargins(0, 0, 0, 0))
        self.setFixedHeight(res.trend_height)
        self.setMinimumWidth(res.trend_width)


class TimeRange(Widget):
    notifyTimeRangeFixed = Signal(pd.Timestamp, pd.Timestamp)

    def __init__(self, res: AppRes):
        super().__init__()
        self.res = res
        self.dt1 = None
        self.dt2 = None

        layout = HBoxLayout()
        self.setLayout(layout)

        self.t_start = t_start = LabelTime()
        layout.addWidget(t_start)

        t_separator = Label("~")
        layout.addWidget(t_separator)

        self.t_end = t_end = LabelTime()
        layout.addWidget(t_end)

        self.pin = but_pin = Button()
        but_pin.setIcon(QIcon(os.path.join(res.dir_image, "pin.png")))
        but_pin.setCheckable(True)
        but_pin.setChecked(False)
        but_pin.toggled.connect(self.on_fix_selection)
        layout.addWidget(but_pin)

    def clearTimeRange(self):
        self.dt1 = None
        self.dt2 = None
        self.t_start.setText("")
        self.t_end.setText("")

    def setTimeRange(self, dt1: pd.Timestamp, dt2: pd.Timestamp):
        self.dt1 = dt1
        self.dt2 = dt2
        self.t_start.setText(dt1.strftime("%H:%M:%S"))
        self.t_end.setText(dt2.strftime("%H:%M:%S"))

    def on_fix_selection(self, state: bool):
        if state:
            self.notifyTimeRangeFixed.emit(self.dt1, self.dt2)
        else:
            self.clearTimeRange()


