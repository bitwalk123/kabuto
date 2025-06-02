from PySide6.QtCore import QMargins, Qt
from PySide6.QtWidgets import QLCDNumber, QLabel, QTextEdit


class Label(QLabel):
    def __init__(self, *args):
        super().__init__(*args)
        self.setContentsMargins(QMargins(0, 0, 0, 0))


class LabelLeft(QLabel):
    def __init__(self, *args):
        super().__init__(*args)
        self.setContentsMargins(QMargins(5, 0, 5, 0))
        self.setAlignment(Qt.AlignmentFlag.AlignLeft)


class LabelRight(QLabel):
    def __init__(self, *args):
        super().__init__(*args)
        self.setContentsMargins(QMargins(5, 0, 5, 0))
        self.setAlignment(Qt.AlignmentFlag.AlignRight)


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


class TextEdit(QTextEdit):
    def __init__(self, *args):
        super().__init__(*args)
