from PySide6.QtWidgets import QPushButton, QStyle, QRadioButton, QButtonGroup


class ButtonBuy(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setCheckable(True)
        self.setStyleSheet("""
            QPushButton {
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
        self.setText("買建")


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
                background-color: #238fe7;
            }
            QPushButton:hover {
                background-color: #7eadec;
            }
        """)
        self.setText("返　　済")


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
        self.setText("売建")


class ButtonSemiAuto(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setCheckable(True)
        self.setFixedHeight(48)
        self.setStyleSheet("""
            QPushButton {
                font-size: 8pt;
                color: black;
                background-color: #fed;
            }
            QPushButton:checked {
                color: white;
                background-color: #432;
            }
        """)
        self.setText("手動エントリ\n自動返済")


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


class ButtonGroup(QButtonGroup):
    def __init__(self, *args):
        super().__init__(*args)
