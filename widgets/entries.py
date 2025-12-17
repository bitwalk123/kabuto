from PySide6.QtCore import QMargins, Qt
from PySide6.QtGui import QDoubleValidator, QIntValidator
from PySide6.QtWidgets import QLineEdit


class Entry(QLineEdit):
    def __init__(self, *args):
        super().__init__(*args)
        self.setStyleSheet("""
            QLineEdit {
                font-family: monospace;
                background-color: white;
                color: black;
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


class EntryFloat(EntryRight):
    def __init__(self, *args):
        super().__init__(*args)
        self.setMinimumWidth(50)
        validator = QDoubleValidator()
        # 科学技術表記（指数表記）を許可しない
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.setValidator(validator)

    def getValue(self) -> float:
        return float(self.text())


class EntryInt(EntryRight):
    def __init__(self, *args):
        super().__init__(*args)
        self.setMinimumWidth(50)
        validator = QIntValidator()
        self.setValidator(validator)

    def getValue(self) -> int:
        return int(self.text())
