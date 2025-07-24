import os

from PySide6.QtCore import Signal
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QToolBar

from structs.res import AppRes


class RhinoToolBar(QToolBar):
    transactionClicked = Signal()

    def __init__(self, res: AppRes):
        super().__init__()
        self.res = res

        self.action_transaction = action_transaction = QAction(
            QIcon(os.path.join(res.dir_image, 'transaction.png')),
            "取引履歴",
            self
        )
        action_transaction.setEnabled(False)
        action_transaction.triggered.connect(self.on_transaction)
        self.addAction(action_transaction)

    def on_transaction(self):
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 「取引履歴」ボタンがクリックされたことを通知
        self.transactionClicked.emit()
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
