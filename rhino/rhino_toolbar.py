import os

from PySide6.QtCore import Signal
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QToolBar, QFileDialog

from structs.res import AppRes


class RhinoToolBar(QToolBar):
    excelSelected = Signal(str)
    playClicked = Signal()
    stopClicked = Signal()
    transactionClicked = Signal()

    def __init__(self, res: AppRes):
        super().__init__()
        self.res = res

        if res.debug:
            action_open = QAction(
                QIcon(os.path.join(res.dir_image, 'excel.png')),
                "Excel ファイルを開く",
                self
            )
            action_open.triggered.connect(self.on_select_excel)
            self.addAction(action_open)

            self.addSeparator()

            action_play = QAction(
                QIcon(os.path.join(res.dir_image, 'play.png')),
                "タイマー開始",
                self
            )
            action_play.triggered.connect(self.on_play)
            self.addAction(action_play)

            action_stop = QAction(
                QIcon(os.path.join(res.dir_image, 'stop.png')),
                "タイマー停止",
                self
            )
            action_stop.triggered.connect(self.on_stop)
            self.addAction(action_stop)


        self.action_transaction = action_transaction = QAction(
            QIcon(os.path.join(res.dir_image, 'transaction.png')),
            "取引履歴",
            self
        )
        action_transaction.setEnabled(False)
        action_transaction.triggered.connect(self.on_transaction)
        self.addAction(action_transaction)

    def on_play(self):
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 「タイマー開始」ボタンがクリックされたことを通知
        self.playClicked.emit()
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
            self.excelSelected.emit(excel_path)
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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
