import os
import re

import pandas as pd
from PySide6.QtCore import QMargins, Signal
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QToolBar

from funcs.tse import get_ticker_name_list
from structs.res import AppRes
from widgets.buttons import Button
from widgets.containers import PadH, Widget
from widgets.dialogs import DlgCSVFileSel
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


class ProfitSimulatorToolbar(QToolBar):
    sendDataFrame = Signal(pd.DataFrame, str, str)
    pattern_code = re.compile(r".*([0-9A-X]{4})_.+\.csv")

    def __init__(self, res: AppRes):
        super().__init__()
        self.res = res
        self.path_csv: str = ""

        # 出力された CSV ファイルを開く
        self.csv = action_open = QAction(
            QIcon(os.path.join(res.dir_image, "csv.png")),
            "CSV ファイルを開く",
            self
        )
        action_open.triggered.connect(self.on_select_technicals)
        self.addAction(action_open)

        pad = PadH()
        self.addWidget(pad)

    def on_select_technicals(self):
        """
        ティックデータを保持した Excel ファイルの選択
        :return:
        """
        # ティックデータ（Excel ファイル）の選択ダイアログ
        dlg_file = DlgCSVFileSel(self.res)
        if dlg_file.exec():
            self.path_csv = dlg_file.selectedFiles()[0]
        else:
            return

        df = pd.read_csv(self.path_csv, index_col=0, parse_dates=True)

        # チャートのタイトル文字列
        d_str = df.index[0].strftime('%Y-%m-%d')
        if m := self.pattern_code.match(self.path_csv):
            code = m.group(1)
        else:
            code = "0000"
        name = get_ticker_name_list([code])[code]
        title = f"{d_str} : {name} ({code})"

        self.sendDataFrame.emit(df, title, self.path_csv)
