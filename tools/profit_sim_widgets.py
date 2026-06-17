import os
import re

import pandas as pd
from PySide6.QtCore import QMargins, Signal
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QDockWidget, QToolBar

from funcs.tse import get_ticker_name_list
from structs.res import AppRes
from widgets.buttons import Button
from widgets.combos import ComboBox
from widgets.containers import PadH, Widget
from widgets.dialogs import DlgOutputFileSel
from widgets.labels import Label, LabelTime
from widgets.layouts import HBoxLayout, VBoxLayout


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
        self.dt1 = pd.Timestamp("1970-01-01")
        self.dt2 = pd.Timestamp("1970-01-01")

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
        self.t_start.setText("")
        self.t_end.setText("")

    def setTimeRange(self, dt1: pd.Timestamp, dt2: pd.Timestamp):
        self.dt1 = dt1
        self.dt2 = dt2
        self.t_start.setText(dt1.strftime("%H:%M:%S"))
        self.t_end.setText(dt2.strftime("%H:%M:%S"))
        # print(self.t_start.width())

    def on_fix_selection(self, state: bool):
        if state:
            self.notifyTimeRangeFixed.emit(self.dt1, self.dt2)


class ProfitSimulatorDock(QDockWidget):
    requestClearSelection = Signal()

    def __init__(self, res: AppRes):
        super().__init__()
        self.res = res
        self.dt1 = pd.Timestamp("1970-01-01")
        self.dt2 = pd.Timestamp("1970-01-01")
        self.df = pd.DataFrame()

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
        self.dt1 = dt1
        self.dt2 = dt2
        self.requestClearSelection.emit()

    def setDataFrame(self, df: pd.DataFrame):
        self.df = df

    def setTimeRange(self, dt1: pd.Timestamp, dt2: pd.Timestamp):
        self.trange.setTimeRange(dt1, dt2)


class ProfitSimulatorToolbar(QToolBar):
    sendDataFrame = Signal(pd.DataFrame, str)
    pattern_code = re.compile(r".*([0-9A-X]{4})_.+\.csv")

    def __init__(self, res: AppRes):
        super().__init__()
        self.res = res

        # 出力された CSV ファイルを開く
        self.csv = action_open = QAction(
            QIcon(os.path.join(res.dir_image, "csv.png")),
            "CSV ファイルを開く",
            self
        )
        action_open.triggered.connect(self.on_select_output)
        self.addAction(action_open)

        pad = PadH()
        self.addWidget(pad)

    def on_select_output(self):
        """
        ティックデータを保持した Excel ファイルの選択
        :return:
        """
        # ティックデータ（Excel ファイル）の選択ダイアログ
        dlg_file = DlgOutputFileSel(self.res)
        if dlg_file.exec():
            path_csv = dlg_file.selectedFiles()[0]
        else:
            return

        if m := self.pattern_code.match(path_csv):
            code = m.group(1)
        else:
            code = "0000"
        name = get_ticker_name_list([code])[code]
        title = f"{name} ({code})"

        df = pd.read_csv(path_csv, index_col=0, parse_dates=True)

        self.sendDataFrame.emit(df, title)
