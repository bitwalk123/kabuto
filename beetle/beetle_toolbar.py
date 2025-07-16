import datetime
import os

from PySide6.QtCore import Signal
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (
    QFileDialog,
    QStyle,
    QToolBar,
)

from structs.res import AppRes
from widgets.buttons import ButtonGroup, RadioButtonInt
from widgets.containers import PadH
from widgets.labels import Label, LCDTime


class ToolBar(QToolBar):
    aboutClicked = Signal()
    excelSelected = Signal(str)
    playClicked = Signal()
    saveClicked = Signal()
    spotTradeClicked = Signal()
    stopClicked = Signal()
    transactionClicked = Signal()
    timerIntervalChanged = Signal(int)

    def __init__(self, res: AppRes):
        super().__init__()
        self.res = res

        if res.debug:
            action_open = QAction(
                self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon),
                "Excel ファイルを開く",
                self
            )
            action_open.triggered.connect(self.on_select_excel)
            self.addAction(action_open)

            self.addSeparator()

            rb_a = RadioButtonInt("10倍速")
            rb_a.toggle()
            rb_a.setValue(100)
            self.addWidget(rb_a)

            rb_b = RadioButtonInt("100倍速")
            rb_b.setValue(10)
            self.addWidget(rb_b)

            self.rb_group = rb_group = ButtonGroup()
            rb_group.addButton(rb_a)
            rb_group.addButton(rb_b)
            rb_group.buttonToggled.connect(self.radiobutton_changed)

            self.addSeparator()

            action_play = QAction(
                self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay),
                "タイマー開始",
                self
            )
            action_play.triggered.connect(self.on_play)
            self.addAction(action_play)

            action_stop = QAction(
                self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop),
                "タイマー停止",
                self
            )
            action_stop.triggered.connect(self.on_stop)
            self.addAction(action_stop)
        # --- debug ここまで ---

        self.action_transaction = action_transaction = QAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogListView),
            "取引履歴",
            self
        )
        action_transaction.setEnabled(False)
        action_transaction.triggered.connect(self.on_transaction)
        self.addAction(action_transaction)

        action_save = QAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton),
            "データを保存する",
            self
        )
        action_save.triggered.connect(self.on_save)
        self.addAction(action_save)

        action_pig = QAction(
            QIcon(os.path.join(res.dir_image, 'pig.png')),
            "現物取引",
            self
        )
        action_pig.triggered.connect(self.on_spot_trade)
        self.addAction(action_pig)

        action_info = QAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxInformation),
            "アプリケーションの情報",
            self
        )
        action_info.triggered.connect(self.on_about)
        self.addAction(action_info)

        hpad = PadH()
        self.addWidget(hpad)

        lab_time = Label("システム時刻 ")
        self.addWidget(lab_time)

        self.lcd_time = lcd_time = LCDTime()
        self.addWidget(lcd_time)

    def on_about(self):
        """
        "アプリケーションの情報" ボタンをクリックした時のシグナル
        :return:
        """
        self.aboutClicked.emit()

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
            self.excelSelected.emit(excel_path)
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def on_play(self):
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 「タイマー開始」ボタンがクリックされたことを通知
        self.playClicked.emit()
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def on_save(self):
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 「データを保存する」ボタンがクリックされたことを通知
        self.saveClicked.emit()
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def on_spot_trade(self):
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 「現物取引」ボタンがクリックされたことを通知
        self.spotTradeClicked.emit()
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def on_stop(self):
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 「タイマー停止」ボタンがクリックされたことを通知
        self.stopClicked.emit()
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def on_transaction(self):
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 「取引履歴」ボタンがクリックされたことを通知
        self.transactionClicked.emit()
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def updateTime(self, ts: float):
        dt = datetime.datetime.fromtimestamp(ts)
        self.lcd_time.display(f"{dt.hour:02}:{dt.minute:02}:{dt.second:02}")

    def radiobutton_changed(self, rb: RadioButtonInt, state: bool):
        if state:
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # 🧿 倍速設定（タイマー間隔）が変更されたことの通知
            self.timerIntervalChanged.emit(rb.getValue())
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def set_transaction(self):
        """
        取引履歴の表示ボタンを Enable にする
        :param df:
        :return:
        """
        self.action_transaction.setEnabled(True)
