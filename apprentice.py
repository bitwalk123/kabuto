import logging
import sys

import xlwings as xw
from PySide6.QtGui import QCloseEvent

if sys.platform == "win32":
    from pywintypes import com_error

from PySide6.QtCore import Qt, QObject, QThread
from PySide6.QtWidgets import (
    QApplication,
    QGridLayout,
    QPushButton,
    QWidget,
)

from funcs.logs import setup_logging


class RSSWorker(QObject):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        if sys.platform == "win32":
            self.wb = xw.Book("collector.xlsm")
        else:
            self.wb = None


class Apprentice(QWidget):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        # Thread for xlwings
        self.thread = QThread(self)

        # GUI
        layout = QGridLayout()
        layout.setSpacing(20)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.setLayout(layout)
        but_sell = QPushButton("売　建")
        layout.addWidget(but_sell, 0, 0)
        but_buy = QPushButton("買　建")
        layout.addWidget(but_buy, 0, 1)
        but_repay = QPushButton("返　　済")
        layout.addWidget(but_repay, 1, 0, 1, 2)

    def closeEvent(self, event: QCloseEvent):
        event.accept()


def main():
    app = QApplication(sys.argv)
    hello = Apprentice()
    hello.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main_logger = setup_logging()
    main()
