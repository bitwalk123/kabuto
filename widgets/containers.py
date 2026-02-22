from PySide6.QtCore import QMargins, Qt
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import (
    QFrame,
    QMainWindow,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QWidget,
)


class Frame(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameStyle(
            QFrame.Shape.StyledPanel | QFrame.Shadow.Plain
        )
        self.setLineWidth(1)


class FrameSunken(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameStyle(
            QFrame.Shape.WinPanel | QFrame.Shadow.Sunken
        )
        self.setLineWidth(2)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setContentsMargins(QMargins(0, 0, 0, 0))


class NarrowLine(QFrame):
    def __init__(self):
        super().__init__()
        self.setContentsMargins(QMargins(0, 0, 0, 0))
        self.setFixedHeight(2)
        self.setLineWidth(0)
        self.setFrameStyle(
            QFrame.Shape.Panel | QFrame.Shadow.Plain
        )
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum
        )
        palette = self.palette()
        self.background_default = palette.color(QPalette.ColorRole.Window)


class IndicatorBuySell(NarrowLine):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(5)
        self.setLineWidth(2)
        self.setFrameStyle(
            QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken
        )

    def setDefault(self) -> None:
        self.setStyleSheet("")
        self.setPalette(self.background_default)

    def setBuy(self) -> None:
        self.setStyleSheet("QFrame{background-color: magenta;}")

    def setSell(self) -> None:
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


class TabWidget(QTabWidget):
    def __init__(self):
        super().__init__()
        self.setContentsMargins(QMargins(0, 0, 0, 0))
        self.setStyleSheet(
            """
            QTabWidget {
                font-family: monospace;
                font-size: 9pt;
            }
            """
        )


class Widget(QWidget):
    def __init__(self):
        super().__init__()
        self.setContentsMargins(QMargins(0, 0, 0, 0))


class ScrollArea(QScrollArea):
    def __init__(self):
        super().__init__()
        self.setContentsMargins(QMargins(0, 0, 0, 0))
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
