import os

import pandas as pd
from PySide6.QtCore import QMargins, Signal
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QDockWidget, QToolBar

from structs.res import AppRes
from widgets.combos import ComboBox
from widgets.containers import PadH, Widget
from widgets.dialogs import DlgOutputFileSel
from widgets.labels import Label, LabelTime
from widgets.layouts import VBoxLayout


class BaseWidget(Widget):
    def __init__(self, res: AppRes):
        super().__init__()
        self.setContentsMargins(QMargins(0, 0, 0, 0))
        self.setFixedHeight(res.trend_height)
        self.setMinimumWidth(res.trend_width)


class ProfitSimulatorDock(QDockWidget):
    def __init__(self, res: AppRes):
        super().__init__()
        self.res = res
        panel = ProfitSimulatorPanel(res)
        self.setWidget(panel)


class ProfitSimulatorPanel(Widget):
    def __init__(self, res: AppRes):
        super().__init__()
        self.res = res
        layout = VBoxLayout()
        self.setLayout(layout)

        self.combo = combo = ComboBox()
        combo.setFixedWidth(200)
        layout.addWidget(combo)


class ProfitSimulatorToolbar(QToolBar):
    sendDataFrame = Signal(pd.DataFrame)
    requestClearSelection = Signal()
    requestSelectorActive = Signal(bool)

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

        self.t_start = t_start = LabelTime()
        self.addWidget(t_start)

        t_separator = Label("~")
        self.addWidget(t_separator)

        self.t_end = t_end = LabelTime()
        self.addWidget(t_end)

        self.pin = action_pin = QAction(
            QIcon(os.path.join(res.dir_image, "pin.png")),
            "選択範囲を確定",
            self
        )
        action_pin.setCheckable(True)
        action_pin.toggled.connect(self.on_fix_selection)
        self.addAction(action_pin)

    def clearTimeRange(self):
        self.t_start.setText("")
        self.t_end.setText("")

    def on_fix_selection(self, state: bool):
        # print(state)
        if state:
            self.requestClearSelection.emit()
            self.requestSelectorActive.emit(False)
        else:
            self.clearTimeRange()
            self.requestSelectorActive.emit(True)

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

        df = pd.read_csv(path_csv, index_col=0, parse_dates=True)
        # print(df)
        # print(df.index.dtype)
        self.sendDataFrame.emit(df)

    def setTimeRange(self, dt1: pd.Timestamp, dt2: pd.Timestamp):
        self.t_start.setText(dt1.strftime("%H:%M:%S"))
        self.t_end.setText(dt2.strftime("%H:%M:%S"))
        # print(self.t_start.width())
