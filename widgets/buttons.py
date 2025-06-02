from PySide6.QtWidgets import QPushButton, QStyle, QRadioButton, QButtonGroup


class ButtonSave(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(
            self.style().standardIcon(
                QStyle.StandardPixmap.SP_DialogSaveButton
            )
        )


class ButtonBuy(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setStyleSheet("""
            QPushButton {
                background-color: #ed6286;
            }
            QPushButton:hover {
                background-color: #f194a7;
            }
        """)
        self.setText("買建")


class ButtonSell(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setStyleSheet("""
            QPushButton {
                background-color: #0ba596;
            }
            QPushButton:hover {
                background-color: #7bbbb1;
            }
        """)
        self.setText("売建")


class ButtonRepay(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setStyleSheet("""
            QPushButton {
                background-color: #238fe7;
            }
            QPushButton:hover {
                background-color: #7eadec;
            }
        """)
        self.setText("返　　済")


class RadioButton(QRadioButton):
    def __init__(self, *args):
        super().__init__(*args)


class ButtonGroup(QButtonGroup):
    def __init__(self, *args):
        super().__init__(*args)
