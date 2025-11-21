from PySide6.QtCore import QMargins
from PySide6.QtWidgets import (
    QLabel,
    QProgressBar,
    QStatusBar,
)

from structs.res import AppRes
from widgets.labels import LabelRight, LabelPrice


class AppStatusBar(QStatusBar):
    def __init__(self, res: AppRes):
        super().__init__()
        self.res = res
        self.pbar = pbar = QProgressBar()
        self.addPermanentWidget(pbar)  # 永続的に表示
        self.lab_status = lab_status = QLabel('準備完了')
        self.lab_status.setContentsMargins(QMargins(5, 0, 0, 0))
        self.lab_status.setStyleSheet(
            """
            QLabel {
                font-size: 8pt;
            }
            """
        )
        self.addWidget(lab_status)

    def setText(self, msg: str):
        self.lab_status.setText(msg)

    def setValue(self, x: int):
        self.pbar.setValue(x)  # エラー時はプログレスバーをリセット


class StatusBar(QStatusBar):
    def __init__(self, res: AppRes):
        super().__init__()
        self.res = res


class TotalBar(QStatusBar):
    def __init__(self):
        super().__init__()
        lab_total = LabelRight("合計収益")
        self.addWidget(lab_total, stretch=1)

        self.lab_price = lab_price = LabelPrice()
        self.addWidget(lab_price)

    def setTotal(self, price: float):
        self.lab_price.setPrice(price)
