import sys

from PySide6.QtWidgets import QApplication, QWidget, QGridLayout, QPushButton


class SampleButtons(QWidget):
    def __init__(self):
        super().__init__()
        layout = QGridLayout()
        self.setLayout(layout)

        but_sell = QPushButton("売　建")
        but_sell.setStyleSheet("""
        QPushButton {
            font-family: monospace;
            background-color: #065;
            color: white;
        }
        """)
        layout.addWidget(but_sell, 0, 0)

        but_buy = QPushButton("買　建")
        but_buy.setStyleSheet("""
        QPushButton {
            font-family: monospace;
            background-color: #a24;
            color: white;
        }
        """)
        layout.addWidget(but_buy, 0, 1)

        but_repay = QPushButton("返　　済")
        but_repay.setStyleSheet("""
        QPushButton {
            font-family: monospace;
            background-color: #039;
            color: white;
        }
        """)
        layout.addWidget(but_repay, 1, 0, 1, 2)

def main():
    app = QApplication(sys.argv)
    win = SampleButtons()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
