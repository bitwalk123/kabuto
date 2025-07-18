from PySide6.QtCore import QMargins, Qt
from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import QSizePolicy, QWidget, QFrame


class Frame(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameStyle(
            QFrame.Shape.StyledPanel | QFrame.Shadow.Plain
        )
        self.setLineWidth(1)


class IndicatorBuySell(QFrame):
    def __init__(self):
        super().__init__()
        self.setContentsMargins(QMargins(0, 0, 0, 0))
        self.setFrameStyle(
            QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken
        )
        self.setLineWidth(2)
        self.setFixedHeight(5)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum
        )
        palette = self.palette()
        self.background_default = palette.color(QPalette.ColorRole.Window)
        # print(f"Default background color (RGB): {self.background_default.getRgb()}")

    def setDefault(self):
        self.setStyleSheet("")
        self.setPalette(self.background_default)

    def setBuy(self):
        self.setStyleSheet("QFrame{background-color: magenta;}")

    def setSell(self):
        self.setStyleSheet("QFrame{background-color: cyan;}")


class PadH(QWidget):
    def __init__(self):
        super().__init__()
        self.setContentsMargins(0, 0, 0, 0)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred
        )


class PadV(QWidget):
    def __init__(self):
        super().__init__()
        self.setContentsMargins(0, 0, 0, 0)
        self.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Expanding
        )


class Widget(QWidget):
    def __init__(self):
        super().__init__()
        self.setContentsMargins(QMargins(0, 0, 0, 0))
