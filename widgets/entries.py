from PySide6.QtCore import QMargins, Qt
from PySide6.QtWidgets import QLineEdit


class Entry(QLineEdit):
    def __init__(self, *args):
        super().__init__(*args)
        self.setStyleSheet("""
            QLineEdit {
                font-family: monospace;
                background-color: white;
            }
        """)
        self.setContentsMargins(QMargins(0, 0, 0, 0))


class EntryAddress(Entry):
    def __init__(self, *args):
        super().__init__(*args)
        self.setReadOnly(True)
        self.setFixedWidth(180)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.address: str = self.text()

    def getAddress(self) -> str:
        return self.address

    def setAddress(self, address: str):
        self.address = address
        self.setText(address)

    def setClear(self):
        self.setText("")


class EntryPort(Entry):
    def __init__(self, *args):
        super().__init__(*args)
        self.setReadOnly(True)
        self.setFixedWidth(70)
        self.setAlignment(Qt.AlignmentFlag.AlignRight)
        if self.text() != "":
            self.port: int = int(self.text())

    def getPort(self) -> int:
        return self.port

    def setPort(self, port: int):
        self.port = port
        self.setText(str(port))

    def setClear(self):
        self.setText("")


class EntryRight(Entry):
    def __init__(self, *args):
        super().__init__(*args)
        self.setMinimumWidth(100)
        self.setAlignment(Qt.AlignmentFlag.AlignRight)


class EntryRightNarrow(EntryRight):
    def __init__(self, *args):
        super().__init__(*args)
        self.setMinimumWidth(50)
