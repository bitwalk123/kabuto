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
            self.clear_logs = wb.macro("ClearLogs")
            self.do_buy = wb.macro("DoBuy")
            self.do_sell = wb.macro("DoSell")
            self.do_repay = wb.macro("DoRepay")
            # 古いログをクリア
            self.macro_clear_logs()

    def macro_clear_logs(self):
        if self.wb is None:
            print("doBuy: 非Windows 上で実行されました。")
            return
        try:
            self.clear_logs()
            self.logger.info("ClearLogs completed")
        except com_error as e:
            self.logger.error(f"ClearLogs failed: {e}")
        except Exception as e:
            self.logger.exception(f"Unexpected error in ClearLogs: {e}")

    def macro_do_buy(self, code: str):
        if self.wb is None:
            print("doBuy: 非Windows 上で実行されました。")
            return
        try:
            result = self.do_buy(code)
            self.logger.info(f"DoBuy returned {result}")
        except com_error as e:
            self.logger.error(f"DoBuy failed for code={code}: {e}")
            return
        except Exception as e:
            self.logger.exception(f"Unexpected error in DoBuy: {e}")
            return

    def macro_do_sell(self, code: str):
        if self.wb is None:
            print("doSell: 非Windows 上で実行されました。")
            return
        try:
            result = self.do_sell(code)
            self.logger.info(f"DoSell returned {result}")
        except com_error as e:
            self.logger.error(f"DoSell failed for code={code}: {e}")
            return
        except Exception as e:
            self.logger.exception(f"Unexpected error in DoSell: {e}")
            return

    def macro_do_repay(self, code: str):
        if self.wb is None:
            print("doRepay: 非Windows 上で実行されました。")
            return
        try:
            result = self.do_repay(code)
            self.logger.info(f"DoRepay returned {result}")
            return
        except com_error as e:
            self.logger.error(f"DoRepay failed for code={code}: {e}")
        except Exception as e:
            self.logger.exception(f"Unexpected error in DoRepay: {e}")
            return


class Button(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setFixedHeight(50)
        self.setMinimumWidth(100)


class Apprentice(QWidget):
    requestWorkerInit = Signal()
    requestBuy = Signal(str)
    requestSell = Signal(str)
    requestRepay = Signal(str)

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        # self.code = "9432"  # 売買テスト用の銘柄コード
        self.code = "8410"  # 売買テスト用の銘柄コード

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
        self.setWindowTitle("売買テスト")
        layout = QGridLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.setLayout(layout)
        self.but_sell = but_sell = Button("売　建")
        but_sell.clicked.connect(self.request_sell)
        layout.addWidget(but_sell, 0, 0)
        self.but_buy = but_buy = Button("買　建")
        but_buy.clicked.connect(self.request_buy)
        layout.addWidget(but_buy, 0, 1)
        self.but_repay = but_repay = Button("返　　済")
        but_repay.clicked.connect(self.request_repay)
        layout.addWidget(but_repay, 1, 0, 1, 2)
        but_reset = QPushButton("解　　除")
        but_reset.clicked.connect(self.on_reset)
        layout.addWidget(but_reset, 2, 0, 1, 2)
        self.switch_activate(True)

    def closeEvent(self, event: QCloseEvent):
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait()
        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None
        event.accept()

    def on_reset(self):
        self.switch_activate(True)

    def request_buy(self):
        self.switch_deactivate_all()
        self.requestBuy.emit(self.code)
        self.switch_activate(False)

    def request_sell(self):
        self.switch_deactivate_all()
        self.requestSell.emit(self.code)
        self.switch_activate(False)

    def request_repay(self):
        self.switch_deactivate_all()
        self.requestRepay.emit(self.code)
        self.switch_activate(True)

    def switch_deactivate_all(self):
        self.but_buy.setDisabled(True)
        self.but_sell.setDisabled(True)
        self.but_repay.setDisabled(True)

    def switch_activate(self, state: bool):
        self.but_buy.setEnabled(state)
        self.but_sell.setEnabled(state)
        self.but_repay.setDisabled(state)

    def showEvent(self, event):
        super().showEvent(event)
        # 表示後の最終サイズを固定
        self.setFixedSize(self.size())


def main():
    app = QApplication(sys.argv)
    hello = Apprentice()
    hello.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main_logger = setup_logging()
    main()
