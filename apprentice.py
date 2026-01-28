import logging
import sys
import time

import xlwings as xw
from PySide6.QtGui import QCloseEvent

if sys.platform == "win32":
    from pywintypes import com_error

from PySide6.QtCore import (
    QObject,
    QThread,
    Qt,
    Signal, Slot,
)
from PySide6.QtWidgets import (
    QApplication,
    QGridLayout,
    QPushButton,
    QWidget,
)

from funcs.logs import setup_logging


class RSSWorker(QObject):
    sendResult = Signal(bool)

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.max_retries = 3
        self.wb = None
        self.clear_logs = None
        self.do_buy = None
        self.do_sell = None
        self.do_repay = None
        self.is_position_present = None

    @Slot()
    def initWorker(self):
        if sys.platform == "win32":
            # Excel ファイルが既に開いていることが前提
            self.wb = wb = xw.books["collector.xlsm"]
            self.clear_logs = wb.macro("ClearLogs")
            self.do_buy = wb.macro("DoBuy")
            self.do_sell = wb.macro("DoSell")
            self.do_repay = wb.macro("DoRepay")
            self.is_position_present = wb.macro("IsPositionPresent")
            # 古いログをクリア
            self.macro_clear_logs()

    @Slot()
    def macro_clear_logs(self):
        if sys.platform != "win32":
            self.logger.info(f"{__name__} ClearLogs: 非Windows 上では実行できません。")
            return
        try:
            self.clear_logs()
            self.logger.info(f"{__name__} ClearLogs completed")
        except com_error as e:
            self.logger.error(f"{__name__} ClearLogs failed: {e}")
        except Exception as e:
            self.logger.exception(f"{__name__} Unexpected error in ClearLogs: {e}")

    @Slot(str)
    def macro_do_buy(self, code: str):
        if sys.platform != "win32":
            self.logger.info(f"{__name__} doBuy: 非Windows 上では実行できません。")
            self.sendResult.emit(True)
            return
        try:
            result = self.do_buy(code)
            self.logger.info(f"{__name__} DoBuy returned {result}")
        except com_error as e:
            self.logger.error(f"{__name__} DoBuy failed for code={code}: {e}")
            self.sendResult.emit(False)
            return
        except Exception as e:
            self.logger.exception(f"{__name__} Unexpected error in DoBuy: {e}")
            self.sendResult.emit(False)
            return

        # 注文結果が False の場合はここで終了
        if not result:
            self.sendResult.emit(False)
            return
        # 約定後、買建では建玉一覧に銘柄コードあり (True)
        expected_state = True
        # 約定確認
        self.confirm_execution(code, expected_state)

    @Slot(str)
    def macro_do_sell(self, code: str):
        if sys.platform != "win32":
            self.logger.info(f"{__name__} doSell: 非Windows 上では実行できません。")
            self.sendResult.emit(True)
            return
        try:
            result = self.do_sell(code)
            self.logger.info(f"{__name__} DoSell returned {result}")
        except com_error as e:
            self.logger.error(f"{__name__} DoSell failed for code={code}: {e}")
            self.sendResult.emit(False)
            return
        except Exception as e:
            self.logger.exception(f"{__name__} Unexpected error in DoSell: {e}")
            self.sendResult.emit(False)
            return

        # 注文結果が False の場合はここで終了
        if not result:
            self.sendResult.emit(False)
            return
        # 約定後、売建では建玉一覧に銘柄コードあり (True)
        expected_state = True
        # 約定確認
        self.confirm_execution(code, expected_state)

    @Slot(str)
    def macro_do_repay(self, code: str):
        if sys.platform != "win32":
            self.logger.info(f"{__name__} doRepay: 非Windows 上では実行できません。")
            self.sendResult.emit(True)
            return
        try:
            result = self.do_repay(code)
            self.logger.info(f"{__name__} DoRepay returned {result}")
        except com_error as e:
            self.logger.error(f"{__name__} DoRepay failed for code={code}: {e}")
            self.sendResult.emit(False)
            return
        except Exception as e:
            self.logger.exception(f"{__name__} Unexpected error in DoRepay: {e}")
            self.sendResult.emit(False)
            return

        # 注文結果が False の場合はここで終了
        if not result:
            self.sendResult.emit(False)
            return
        # 約定後、返済では建玉一覧に銘柄コードなし (False)
        expected_state = False
        # 約定確認
        self.confirm_execution(code, expected_state)

    def confirm_execution(self, code: str, expected_state: bool):
        # 約定確認
        for attempt in range(self.max_retries):
            time.sleep(0.5)  # 0.5秒
            try:
                current = bool(self.is_position_present(code)) # 論理値が返ってくるはずだけど保険に
                if current == expected_state:
                    self.logger.info(f"{__name__} 約定が反映されました (attempt {attempt + 1}).")
                    self.sendResult.emit(True)
                    return
                else:
                    self.logger.info(
                        f"{__name__} 約定未反映 (attempt {attempt + 1}): "
                        f"is_position_present={current}, expected={expected_state}"
                    )
            except com_error as e:
                self.logger.error(f"{__name__} IsPositionPresent failed for code={code}: {e}")
                self.logger.info(f"{__name__} retrying... (Attempt {attempt + 1}/{self.max_retries})")
            except Exception as e:
                self.logger.exception(f"{__name__} Unexpected error in IsPositionPresent: {e}")

        # self.max_retries 回確認しても変化なし → 注文未反映
        self.logger.info(f"{__name__} 約定を確認できませんでした。")
        self.sendResult.emit(False)


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
        self.flag_next_status = None

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
        worker.sendResult.connect(self.receive_result)
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

    @Slot(bool)
    def receive_result(self, status: bool):
        if self.flag_next_status is None:
            # 初期状態で誤って呼ばれた場合の保険
            self.switch_activate(True)
            return
        if status:
            self.switch_activate(self.flag_next_status)
        else:
            self.switch_activate(not self.flag_next_status)

    def request_buy(self):
        self.switch_deactivate_all()
        self.flag_next_status = False
        self.requestBuy.emit(self.code)

    def request_sell(self):
        self.switch_deactivate_all()
        self.flag_next_status = False
        self.requestSell.emit(self.code)

    def request_repay(self):
        self.switch_deactivate_all()
        self.flag_next_status = True
        self.requestRepay.emit(self.code)

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
