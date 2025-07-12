from PySide6.QtCore import QMargins, Qt
from PySide6.QtWidgets import QLineEdit


class Entry(QLineEdit):
    def __init__(self, *args):
        super().__init__(*args)
        self.setStyleSheet("""
            QLabel {
                font-family: monospace;
            }
        """)
        self.setContentsMargins(QMargins(0, 0, 0, 0))


class EntryAddress(Entry):
    def __init__(self, *args):
        super().__init__(*args)
        self.setReadOnly(True)
        self.setFixedWidth(150)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def setAddress(self, address: str):
        self.setText(address)

    def setClear(self):
        self.setText("")


class EntryPort(Entry):
    def __init__(self, *args):
        super().__init__(*args)
        self.setReadOnly(True)
        self.setFixedWidth(60)
        self.setAlignment(Qt.AlignmentFlag.AlignRight)

    def setPort(self, port: int):
        self.setText(str(port))

    def setClear(self):
        self.setText("")
