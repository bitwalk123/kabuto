import sys

import xlwings as xw
from PySide6.QtCore import (
    QObject,
    QThread,
    Signal,
    Slot,
)
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class ExcelWorker(QObject):
    def __init__(self):
        super().__init__()
        self.sheet = None

    @Slot()
    def initWorker(self):
        wb = xw.Book()  # 新しいワークブックを開く
        self.sheet = wb.sheets["Sheet1"]  # シート・オブジェクトをインスタンス化

    @Slot(str)
    def excel_write(self, msg: str):
        self.sheet["A1"].value = msg  # 値の書き込み


class SampleXlwings(QWidget):
    requestWorkerInit = Signal()
    requestWrite = Signal(str)

    def __init__(self):
        super().__init__()
        # xlwings用スレッド
        self.thread = thread = QThread()
        self.worker = worker = ExcelWorker()
        worker.moveToThread(thread)
        thread.started.connect(self.requestWorkerInit.emit)
        self.requestWorkerInit.connect(worker.initWorker)
        self.requestWrite.connect(worker.excel_write)
        thread.start()
        # GUI
        layout = QVBoxLayout()
        self.setLayout(layout)
        but = QPushButton("新規 Excel へ文字列を書き込む")
        but.clicked.connect(self.request_write)
        layout.addWidget(but)

    def closeEvent(self, event: QCloseEvent):
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait()
        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None
        event.accept()

    @Slot()
    def request_write(self):
        msg = "Python から書き込みました。"
        self.requestWrite.emit(msg)


def main():
    app = QApplication(sys.argv)
    win = SampleXlwings()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
