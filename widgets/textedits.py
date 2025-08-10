from PySide6.QtWidgets import QTextEdit


class MultilineLog(QTextEdit):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("QTextEdit {font-family: monospace;}")
        self.setReadOnly(True)  # Set it to read-only for history
