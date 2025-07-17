import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
)

from widgets.containers import Widget
from widgets.docks import DockWidget
from widgets.labels import LabelSmall, LCDNumber
from widgets.layouts import VBoxLayout


class DispPrice(Widget):
    def __init__(self, title: str):
        super().__init__()

        vbox = VBoxLayout()
        self.setLayout(vbox)

        lab_price = LabelSmall(title)
        vbox.addWidget(lab_price)
        self.lcd_price = lcd_price = LCDNumber(self)
        vbox.addWidget(lcd_price)


class DockTemplate(DockWidget):
    def __init__(self, title: str):
        super().__init__(title)
        # 現在株価表示
        price = DispPrice("現在株価")
        self.layout.addWidget(price)


class Example(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dock Template")
        ticker = "7011"
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
