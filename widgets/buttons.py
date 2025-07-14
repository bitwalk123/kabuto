import os

from PySide6.QtCore import QMargins
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QButtonGroup,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QStyle,
)

from structs.res import AppRes


class Button(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Preferred
        )
        self.setStyleSheet("""
            QPushButton {
                font-family: monospace;
            }
        """)


class ButtonBuy(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setCheckable(True)
        self.setStyleSheet("""
            QPushButton {
                font-size: 8pt;
                background-color: #ed6286;
            }
            QPushButton:hover {
                background-color: #f194a7;
            }
            QPushButton:disabled {
                background-color: #d75879;
            }
            QPushButton:disabled:checked {
                background-color: #d75879;
                color: white;
            }
        """)
        self.setText("買　建")


class ButtonList(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(
            self.style().standardIcon(
                QStyle.StandardPixmap.SP_FileDialogListView
            )
        )


class ButtonRepay(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setCheckable(True)
        self.setStyleSheet("""
            QPushButton {
                font-size: 8pt;
                background-color: #238fe7;
            }
            QPushButton:hover {
                background-color: #7eadec;
            }
        """)
        self.setText("返　　済")


class ButtonConf(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setContentsMargins(QMargins(0, 0, 0, 0))
        self.setIcon(
            self.style().standardIcon(
                QStyle.StandardPixmap.SP_FileDialogDetailedView
            )
        )


class ButtonGroup(QButtonGroup):
    def __init__(self, *args):
        super().__init__(*args)


class ButtonPig(QPushButton):
    def __init__(self, res: AppRes):
        super().__init__()
        self.setContentsMargins(QMargins(0, 0, 0, 0))
        imgname = os.path.join(res.dir_image, 'pig.png')
        self.setIcon(QIcon(imgname))


class ButtonSave(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(
            self.style().standardIcon(
                QStyle.StandardPixmap.SP_DialogSaveButton
            )
        )


class ButtonSell(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setCheckable(True)
        self.setStyleSheet("""
            QPushButton {
                font-size: 8pt;
                background-color: #0ba596;
            }
            QPushButton:hover {
                background-color: #7bbbb1;
            }
            QPushButton:disabled {
                background-color: #099588;
            }
            QPushButton:disabled:checked {
                background-color: #099588;
                color: white;
            }
        """)
        self.setText("売　建")


class ButtonTicker(QPushButton):
    def __init__(self, code: str, name: str):
        super().__init__()
        self.code = code
        self.name = name
        self.setText(f"{code} {name}")
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred
        )
        self.setStyleSheet("""
            QPushButton {
                font-size: small;
                font-family: monospace;
                padding-left: 0.5em;
                padding-right: 0.5em;
                text-align: left;
            }
        """)
        self.setCheckable(True)
        self.setAutoExclusive(True)


    def getCode(self) -> str:
        return self.code

    def getName(self) -> str:
        return self.name


class ButtonSemiAuto(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setCheckable(True)
        self.setStyleSheet("""
            QPushButton {
                font-size: 8pt;
                font-weight: bold;
                color: black;
                background-color: #fed;
            }
            QPushButton:checked {
                color: white;
                background-color: #432;
            }
            QPushButton:disabled {
                color: #888;
                background-color: #ccc;
            }
        """)
        self.setText("Semi AUTO")
        self.setToolTip("セミオート")


class RadioButton(QRadioButton):
    def __init__(self, *args):
        super().__init__(*args)


class RadioButtonInt(QRadioButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.value: int = 0

    def setValue(self, val: int):
        self.value = val

    def getValue(self) -> int:
        return self.value


class ToggleButtonAutoPilot(QPushButton):
    def __init__(self, res: AppRes):
        super().__init__()
        self.setContentsMargins(QMargins(0, 0, 0, 0))
        self.setCheckable(True)
        imgname = os.path.join(res.dir_image, "autopilot.png")
        self.setIcon(QIcon(imgname))
