from PySide6.QtWidgets import QLCDNumber


class LCDNumber(QLCDNumber):
    def __init__(self, *args):
        super().__init__(*args)
        self.setFixedWidth(160)
        self.setFixedHeight(24)
        self.setDigitCount(12)
        self.display('0.0')
