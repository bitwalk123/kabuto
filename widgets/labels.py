from PySide6.QtCore import QMargins, Qt
from PySide6.QtWidgets import QLCDNumber, QLabel, QPlainTextEdit, QFrame, QSizePolicy


class Label(QLabel):
    def __init__(self, *args):
        super().__init__(*args)
        self.setStyleSheet("""
            QLabel {
                font-family: monospace;
            }
        """)
        self.setContentsMargins(QMargins(0, 0, 0, 0))


class LabelLeft(QLabel):
    def __init__(self, *args):
        super().__init__(*args)
        self.setStyleSheet("""
            QLabel {
                font-family: monospace;
            }
        """)
        self.setContentsMargins(QMargins(5, 1, 5, 1))
        self.setAlignment(Qt.AlignmentFlag.AlignLeft)


class LabelRight(QLabel):
    def __init__(self, *args):
        super().__init__(*args)
        self.setStyleSheet("""
            QLabel {
                font-family: monospace;
            }
        """)
        self.setContentsMargins(QMargins(5, 1, 5, 1))
        self.setAlignment(Qt.AlignmentFlag.AlignRight)


class LabelPrice(LabelRight):
    def __init__(self, price: float = 0):
        super().__init__()
        self.setFrameStyle(
            QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken
        )
        self.setLineWidth(2)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred
        )
        self.setFixedWidth(120)
        self.setStyleSheet("""
            QLabel {
                font-family: monospace;
                background-color: white;
                color: black;
                padding-left: 5px;
                padding-right: 5px;
            }
        """)
        self.setPrice(price)

    def setPrice(self, price: float):
        self.setText(f"{price:,.1f}")


class LCDNumber(QLCDNumber):
    def __init__(self, *args):
        super().__init__(*args)
        self.setFixedWidth(160)
        self.setFixedHeight(24)
        self.setDigitCount(12)
        self.display('0.0')


class LCDTime(QLCDNumber):
    def __init__(self, *args):
        super().__init__(*args)
        self.setFixedWidth(100)
        self.setDigitCount(8)
        self.display('00:00:00')


class PlainTextEdit(QPlainTextEdit):
    def __init__(self, *args):
        super().__init__(*args)
        self.setStyleSheet("""
            QPlainTextEdit {
                border-width: 0;
                border-style: none;
                padding: 0;
            }
        """)
