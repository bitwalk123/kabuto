from PySide6.QtCore import QMargins
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

    def setAddress(self, address: str):
        self.setText(address)


class EntryPort(Entry):
    def __init__(self, *args):
        super().__init__(*args)
        self.setReadOnly(True)

    def setPort(self, port: int):
        self.setText(str(port))
