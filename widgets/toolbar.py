import datetime

from PySide6.QtCore import Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QStyle, QToolBar, QFileDialog

from structs.res import AppRes
from widgets.buttons import RadioButton, ButtonGroup
from widgets.containers import PadH
from widgets.labels import LCDTime, Label


class ToolBar(QToolBar):
    excelSelected = Signal(str)
    playClicked = Signal()
    saveClicked = Signal()
    stopClicked = Signal()

    def __init__(self, res: AppRes):
        super().__init__()
        self.res = res

        if res.debug:
            action_open = QAction(
                self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon),
                'Excel ファイルを開く',
                self
            )
            action_open.triggered.connect(self.on_select_excel)
            self.addAction(action_open)

            self.addSeparator()

            rb_a = RadioButton('10倍速')
            rb_a.toggle()
            self.addWidget(rb_a)

            rb_b = RadioButton('100倍速')
            self.addWidget(rb_b)

            self.rb_group = rb_group = ButtonGroup()
            rb_group.addButton(rb_a)
            rb_group.addButton(rb_b)
            rb_group.buttonToggled.connect(self.radiobutton_changed)

            self.addSeparator()

            action_play = QAction(
                self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay),
                'タイマー開始',
                self
            )
            action_play.triggered.connect(self.on_play)
            self.addAction(action_play)

            action_stop = QAction(
                self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop),
                'タイマー停止',
                self
            )
            action_stop.triggered.connect(self.on_stop)
            self.addAction(action_stop)
        # --- debug ここまで ---

        action_save = QAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton),
            'データを保存する',
            self
        )
        action_save.triggered.connect(self.on_save)
        self.addAction(action_save)

        action_info = QAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxInformation),
            'アプリケーションの情報',
            self
        )
        self.addAction(action_info)

        hpad = PadH()
        self.addWidget(hpad)

        lab_time = Label("現在時刻 ")
        self.addWidget(lab_time)

        self.lcd_time = lcd_time = LCDTime()
        self.addWidget(lcd_time)

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
            self.excelSelected.emit(excel_path)

    def on_play(self):
        self.playClicked.emit()

    def on_save(self):
        self.saveClicked.emit()

    def on_stop(self):
        self.stopClicked.emit()

    def updateTime(self):
        dt = datetime.datetime.now()
        self.lcd_time.display(f"{dt.hour:02}:{dt.minute:02}:{dt.second:02}")

    def radiobutton_changed(self, rb: RadioButton, state: bool):
        if state:
            status = 'オン'
        else:
            status = 'オフ'
        print('「%s」を%sにしました。' % (rb.text(), status))
