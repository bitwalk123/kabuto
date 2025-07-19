import datetime

from PySide6.QtCore import Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QFileDialog,
    QStyle,
    QToolBar,
)

from structs.res import AppRes
from widgets.containers import PadH
from widgets.labels import LCDTime, Label


class ToolBarTransaction(QToolBar):
    transdataSelected = Signal(str)
    saveClicked = Signal()

    def __init__(self, res: AppRes):
        super().__init__()
        self.res = res

        action_save = QAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton),
            "取引履歴を保存する",
            self
        )
        action_save.triggered.connect(self.on_save)
        self.addAction(action_save)

        action_open = QAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon),
            "Excel ファイル（取引履歴）を開く",
            self
        )
        action_open.triggered.connect(self.on_select_excel)
        self.addAction(action_open)

    def on_save(self):
        # ----------------------------------------------
        # 🧿 「取引履歴を保存する」ボタンがクリックされたことを通知
        self.saveClicked.emit()
        # ----------------------------------------------

    def on_select_excel(self):
        excel_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open File",
            self.res.dir_transaction,
            "Excel File (*.xlsx)"
        )
        if excel_path == "":
            return
        else:
            # ----------------------------------
            # 🧿 Excel ファイルが選択されたことの通知
            self.transdataSelected.emit(excel_path)
            # ----------------------------------


class ToolBarVein(QToolBar):
    def __init__(self, res: AppRes):
        super().__init__()
        self.setFixedHeight(32)
        self.res = res

        hpad = PadH()
        self.addWidget(hpad)

        lab_time = Label("システム時刻 ")
        self.addWidget(lab_time)

        self.lcd_time = lcd_time = LCDTime()
        self.addWidget(lcd_time)

    def updateTime(self, ts: float):
        dt = datetime.datetime.fromtimestamp(ts)
        self.lcd_time.display(f"{dt.hour:02}:{dt.minute:02}:{dt.second:02}")
