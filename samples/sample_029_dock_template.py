import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
)

from widgets.containers import Widget
from widgets.docks import DockWidget
from widgets.labels import LCDValueWithTitle, LCDIntWithTitle


class DockTemplate(DockWidget):
    def __init__(self, title: str):
        super().__init__(title)
        # 現在株価
        price = LCDValueWithTitle("現在株価")
        self.layout.addWidget(price)
        # 含み損益
        profit = LCDValueWithTitle("含み損益")
        self.layout.addWidget(profit)
        # 合計収益
        total = LCDValueWithTitle("合計収益")
        self.layout.addWidget(total)
        # EP 更新回数
        epupd = LCDIntWithTitle("EP 更新回数")
        self.layout.addWidget(epupd)


class Example(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dock Template")
        ticker = "Ticker"
        dock = DockTemplate(ticker)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

        base = Widget()
        self.setCentralWidget(base)


def main():
    app = QApplication(sys.argv)
    win = Example()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
