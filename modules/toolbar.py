import datetime
import os
import re

from PySide6.QtCore import Signal
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QFileDialog, QToolBar

from structs.res import AppRes
from widgets.combos import ComboBox
from widgets.containers import PadH
from widgets.labels import Label, LCDTime


class ToolBar(QToolBar):
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


class ToolBarProphet(QToolBar):
    clickedPlay = Signal()

    def __init__(self, res: AppRes):
        super().__init__()
        self.res = res

        self.dir_model = os.path.join(self.res.dir_model, "trained")
        self.dir_collection = self.res.dir_collection

        action_start = QAction(
            QIcon(os.path.join(res.dir_image, 'play.png')),
            "推論の開始",
            self
        )
        action_start.triggered.connect(self.on_start_inference)
        self.addAction(action_start)

        self.addSeparator()

        lab_model = Label("モデル")
        lab_model.setStyleSheet("QLabel {padding: 0 5px 0 5px;}")
        self.addWidget(lab_model)

        self.combo_model = combo_model = ComboBox()
        combo_model.setToolTip("学習済みモデル一覧")
        combo_model.addItems(self.get_trained_models())
        self.addWidget(combo_model)

        self.addSeparator()

        lab_tick = Label("ティックデータ")
        lab_tick.setStyleSheet("QLabel {padding: 0 5px 0 5px;}")
        self.addWidget(lab_tick)

        self.combo_tick = combo_tick = ComboBox()
        combo_tick.setToolTip("ティックデータ一覧")
        combo_tick.addItems(self.get_tick_data())
        self.addWidget(combo_tick)

        self.addSeparator()

        action_update = QAction(
            QIcon(os.path.join(res.dir_image, 'update.png')),
            "ファイル一覧の更新",
            self
        )
        action_update.triggered.connect(self.on_update)
        self.addAction(action_update)

    def get_tick_data(self) -> list[str]:
        """
        ティックデータ一覧の取得
        :return:
        """
        list_tick = sorted(os.listdir(self.dir_collection), reverse=True)
        return list_tick

    def get_trained_models(self) -> list[str]:
        """
        学習済みモデル一覧の取得
        :return:
        """
        list_model = sorted(os.listdir(self.dir_model), reverse=True)
        return list_model

    def getInfo(self) -> dict:
        dict_info = dict()
        model = self.combo_model.currentText()
        pattern = re.compile(r"ppo_([A-Z0-9]{4}).*\.zip")
        if m := pattern.match(model):
            code = m.group(1)
        else:
            code = "unknown"
        dict_info["code"] = code
        path_model = os.path.join(self.dir_model, model)
        dict_info["path_model"] = path_model

        excel = self.combo_tick.currentText()
        path_excel = os.path.join(self.dir_collection, excel)
        dict_info["path_excel"] = path_excel

        return dict_info

    def on_start_inference(self):
        self.clickedPlay.emit()

    def on_update(self):
        pass
