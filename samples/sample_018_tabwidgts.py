import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QTabWidget, QSizePolicy


class Example(QTabWidget):
    def __init__(self):
        super().__init__()
        main = QMainWindow()
        main.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        self.addTab(main, "メイン")
        self.base = base = QWidget()
        main.setCentralWidget(base)

        self.timer = timer = QTimer()
        timer.setInterval(1000)
        timer.timeout.connect(self.on_test)
        timer.start()

    def on_test(self):
        self.base.setFixedSize(800, 600)
        self.timer.stop()
        self.resize(self.base.size())


def main():
    app = QApplication(sys.argv)
    win = Example()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
