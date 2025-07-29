import datetime
import os

from PySide6.QtCore import Signal
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QFileDialog, QToolBar

from structs.res import AppRes
from widgets.containers import PadH
from widgets.labels import Label, LCDTime


class RhinoToolBar(QToolBar):
    clickedAbout = Signal()
    clickedPlay = Signal()
    clickedStop = Signal()
    clickedTransaction = Signal()
    selectedExcelFile = Signal(str)

    def __init__(self, res: AppRes):
        super().__init__()
        self.res = res

        # デバッグ（レビュー）モード時のみ
        if res.debug:
            # Excel ファイルを開く
            action_open = QAction(
                QIcon(os.path.join(res.dir_image, 'excel.png')),
                "Excel ファイルを開く",
                self
            )
            action_open.triggered.connect(self.on_select_excel)
            self.addAction(action_open)

            self.addSeparator()

            # タイマー開始
            action_play = QAction(
                QIcon(os.path.join(res.dir_image, 'play.png')),
                "タイマー開始",
                self
            )
            action_play.triggered.connect(self.on_play)
            self.addAction(action_play)

            # タイマー停止
            action_stop = QAction(
                QIcon(os.path.join(res.dir_image, 'stop.png')),
                "タイマー停止",
                self
            )
            action_stop.triggered.connect(self.on_stop)
            self.addAction(action_stop)

        # 取引履歴
        self.action_transaction = action_transaction = QAction(
            QIcon(os.path.join(res.dir_image, 'transaction.png')),
            "取引履歴",
            self
        )
        action_transaction.setEnabled(False)
        action_transaction.triggered.connect(self.on_transaction)
        self.addAction(action_transaction)

        # このアプリについて
        self.action_about = action_about = QAction(
            QIcon(os.path.join(res.dir_image, "about.png")),
            "このアプリについて",
            self
        )
        action_about.triggered.connect(self.on_about)
        self.addAction(action_about)

        pad = PadH()
        self.addWidget(pad)

        lab_time = Label("システム時刻 ")
        self.addWidget(lab_time)

        self.lcd_time = lcd_time = LCDTime()
        self.addWidget(lcd_time)

    def on_about(self):
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 「このアプリについて」ボタンがクリックされたことを通知
        self.clickedAbout.emit()
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def on_play(self):
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 「タイマー開始」ボタンがクリックされたことを通知
        self.clickedPlay.emit()
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def on_select_excel(self):
        excel_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open File",
            self.res.dir_excel,
            "Excel File (*.xlsx)"
        )
        if excel_path == "":
            return
        else:
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # 🧿 Excel ファイルが選択されたことの通知
            self.selectedExcelFile.emit(excel_path)
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def on_stop(self):
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 「タイマー停止」ボタンがクリックされたことを通知
        self.clickedStop.emit()
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def on_transaction(self):
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 「取引履歴」ボタンがクリックされたことを通知
        self.clickedTransaction.emit()
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def set_transaction(self):
        """
        取引履歴の表示ボタンを Enable にする
        :param df:
        :return:
        """
        self.action_transaction.setEnabled(True)

    def updateTime(self, ts: float):
        dt = datetime.datetime.fromtimestamp(ts)
        self.lcd_time.display(f"{dt.hour:02}:{dt.minute:02}:{dt.second:02}")
