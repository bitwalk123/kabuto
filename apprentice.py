import logging
import sys

import xlwings as xw
from PySide6.QtGui import QCloseEvent

if sys.platform == "win32":
    from pywintypes import com_error

from PySide6.QtCore import (
    QObject,
    QThread,
    Qt,
    Signal,
)
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
        self.wb = None
        self.do_buy = None
        self.do_sell = None
        self.do_repay = None

    def initWorker(self):
        if sys.platform == "win32":
            # Excel ファイルが既に開いていることが前提
            self.wb = wb = xw.books["collector.xlsm"]
            self.do_buy = wb.macro("Module1.DoBuy")
            self.do_sell = wb.macro("Module1.DoSell")
            self.do_repay = wb.macro("Module1.DoRepay")

    def macro_do_buy(self):
        if self.wb is not None:
            self.do_buy()
        else:
            print("doBuy: 非Windows 上で実行されました。")

    def macro_do_sell(self):
        if self.wb is not None:
            self.do_sell()
        else:
            print("doSell: 非Windows 上で実行されました。")

    def macro_do_repay(self):
        if self.wb is not None:
            self.do_repay()
        else:
            print("doRepay: 非Windows 上で実行されました。")


class Apprentice(QWidget):
    requestWorkerInit = Signal()
    requestBuy = Signal()
    requestSell = Signal()
    requestRepay = Signal()

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        # Thread for xlwings
        self.thread = QThread(self)
        self.worker = worker = RSSWorker()
        worker.moveToThread(self.thread)
        self.thread.started.connect(self.requestWorkerInit.emit)
        self.requestWorkerInit.connect(worker.initWorker)
        self.requestBuy.connect(worker.macro_do_buy)
        self.requestSell.connect(worker.macro_do_sell)
        self.requestRepay.connect(worker.macro_do_repay)
        self.thread.start()

        # GUI
        layout = QGridLayout()
        layout.setSpacing(20)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.setLayout(layout)
        but_sell = QPushButton("売　建")
        but_sell.clicked.connect(self.requestSell.emit)
        layout.addWidget(but_sell, 0, 0)
        but_buy = QPushButton("買　建")
        but_buy.clicked.connect(self.requestBuy.emit)
        layout.addWidget(but_buy, 0, 1)
        but_repay = QPushButton("返　　済")
        but_repay.clicked.connect(self.requestRepay.emit)
        layout.addWidget(but_repay, 1, 0, 1, 2)

    def closeEvent(self, event: QCloseEvent):
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait()
        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None
        event.accept()


def main():
    app = QApplication(sys.argv)
    hello = Apprentice()
    hello.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main_logger = setup_logging()
    main()
