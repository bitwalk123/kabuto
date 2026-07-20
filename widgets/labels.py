import time

from PySide6.QtCore import QMargins, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QLCDNumber,
    QSizePolicy,
)

from widgets.containers import Widget
from widgets.layouts import VBoxLayout, HBoxLayout


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


class LabelLeft2(QLabel):
    def __init__(self, *args):
        super().__init__(*args)
        self.setStyleSheet("""
            QLabel {
                font-family: monospace;
            }
        """)
        self.setContentsMargins(QMargins(5, 1, 10, 1))
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)


class LabelRaised(Label):
    def __init__(self, *args):
        super().__init__(*args)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFrameStyle(QFrame.Shape.Panel | QFrame.Shadow.Raised)
        self.setLineWidth(2)


class LabelRaisedLeft(Label):
    def __init__(self, *args):
        super().__init__(*args)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.setFrameStyle(QFrame.Shape.Panel | QFrame.Shadow.Raised)
        self.setLineWidth(2)


class LabelRaisedRight(Label):
    def __init__(self, *args):
        super().__init__(*args)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.setFrameStyle(QFrame.Shape.Panel | QFrame.Shadow.Raised)
        self.setLineWidth(2)


class LabelRight(QLabel):
    def __init__(self, *args):
        super().__init__(*args)
        self.setStyleSheet("""
            QLabel {font-family: monospace;}
        """)
        self.setContentsMargins(QMargins(5, 1, 5, 1))
        self.setAlignment(Qt.AlignmentFlag.AlignRight)


class LabelRightMedium(LabelRight):
    def __init__(self, *args):
        super().__init__(*args)
        font = QFont()
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPointSize(9)
        self.setFont(font)


class LabelRightSmall(LabelRight):
    def __init__(self, *args):
        super().__init__(*args)
        font = QFont()
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPointSize(6)
        self.setFont(font)


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


class LabelSmall(QLabel):
    def __init__(self, *args):
        super().__init__(*args)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum
        )
        self.setFrameStyle(
            QFrame.Shape.WinPanel | QFrame.Shadow.Raised
        )
        self.setLineWidth(1)
        self.setContentsMargins(QMargins(1, 1, 1, 1))
        self.setStyleSheet("""
            QLabel {padding-left: 0.5em;}
        """)
        font = QFont()
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPointSize(6)
        self.setFont(font)


class LabelTitle(QLabel):
    def __init__(self, *args):
        super().__init__(*args)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum
        )
        self.setFrameStyle(
            QFrame.Shape.WinPanel | QFrame.Shadow.Raised
        )
        self.setLineWidth(1)
        self.setContentsMargins(QMargins(1, 1, 1, 1))
        self.setStyleSheet("""
            QLabel {padding-left: 0.5em;}
        """)
        font = QFont()
        font.setStyleHint(QFont.StyleHint.Monospace)
        # font.setPointSize(6)
        self.setFont(font)


class LabelTime(Label):
    def __init__(self, *args):
        super().__init__(*args)
        self.setFixedWidth(80)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)


class LCDInt(QLCDNumber):
    def __init__(self, *args):
        super().__init__(*args)
        self.setSegmentStyle(QLCDNumber.SegmentStyle.Flat)
        self.setDigitCount(12)
        self.display('0')


class LCDNumber(QLCDNumber):
    def __init__(self, *args):
        super().__init__(*args)
        self.setSegmentStyle(QLCDNumber.SegmentStyle.Flat)
        self.setDigitCount(12)
        self.display('0.0')


class LCDTime(QLCDNumber):
    def __init__(self, *args):
        super().__init__(*args)
        self.setSegmentStyle(QLCDNumber.SegmentStyle.Flat)
        self.setDigitCount(8)
        self.display('00:00:00')

    def setTimestamp(self, ts: float):
        self.display(time.strftime("%H:%M:%S", time.localtime(ts)))


class LCDValueWithTitle(Widget):
    def __init__(self, title: str):
        super().__init__()
        # layout
        layout = HBoxLayout()
        self.setLayout(layout)
        # title
        lab_title = LabelTitle(title)
        lab_title.setFixedWidth(80)
        layout.addWidget(lab_title)
        # LCD
        self.lcd_value = lcd_value = LCDNumber(self)
        layout.addWidget(lcd_value)

    def getValue(self) -> float:
        """
        LCD に表示されている数値を取得
        :return:
        """
        return self.lcd_value.value()

    def setValue(self, value: float):
        """
        LCD に数値を表示
        :param value:
        :return:
        """
        self.lcd_value.display(f"{value:.1f}")


class LCDIntWithTitle(Widget):
    def __init__(self, title: str):
        super().__init__()
        # layout
        layout = VBoxLayout()
        self.setLayout(layout)
        # title
        lab_title = LabelSmall(title)
        layout.addWidget(lab_title)
        # LCD
        self.lcd_int = lcd_value = LCDInt(self)
        layout.addWidget(lcd_value)

    def getValue(self) -> int:
        """
        LCD に表示されている数値を取得
        :return:
        """
        return int(self.lcd_int.value())

    def setValue(self, value: int):
        """
        LCD に数値を表示
        :param value:
        :return:
        """
        self.lcd_int.display(f"{value:d}")
