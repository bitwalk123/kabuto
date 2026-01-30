import sys

from PySide6.QtWidgets import QApplication, QWidget, QGridLayout, QPushButton


class SampleButtons(QWidget):
    def __init__(self):
        super().__init__()
        layout = QGridLayout()
        self.setLayout(layout)

        but_sell = QPushButton("売　建")
        but_sell.setDisabled(True)
        but_sell.setStyleSheet("""
        QPushButton {
            font-family: monospace;
            background-color: #065;
            color: white;
        }
        QPushButton:pressed {
            font-family: monospace;
            background-color: #098;
            color: white;
        }
        QPushButton:disabled {
            font-family: monospace;
            background-color: #032;
            color: gray;
        }
        """)
        layout.addWidget(but_sell, 0, 0)

        but_buy = QPushButton("買　建")
        but_buy.setDisabled(True)
        but_buy.setStyleSheet("""
        QPushButton {
            font-family: monospace;
            background-color: #a24;
            color: white;
        }
        QPushButton:pressed {
            font-family: monospace;
            background-color: #d45;
            color: white;
        }
        QPushButton:disabled {
            font-family: monospace;
            background-color: #512;
            color: gray;
        }
        """)
        layout.addWidget(but_buy, 0, 1)

        but_repay = QPushButton("返　　済")
        but_repay.setDisabled(True)
        but_repay.setStyleSheet("""
        QPushButton {
            font-family: monospace;
            background-color: #039;
            color: white;
        }
        QPushButton:pressed {
            font-family: monospace;
            background-color: #07d;
            color: white;
        }
        QPushButton:disabled {
            font-family: monospace;
            background-color: #016;
            color: gray;
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
