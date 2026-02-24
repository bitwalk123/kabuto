from PySide6.QtWidgets import QTextEdit, QPlainTextEdit


class MultilineLog(QTextEdit):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("QTextEdit {font-family: monospace;}")
        self.setReadOnly(True)  # Set it to read-only for history


class PlainTextEdit(QPlainTextEdit):
    def __init__(self, *args):
        super().__init__(*args)
        self.setStyleSheet("""
            QPlainTextEdit {
                border-width: 0;
                border-style: none;
                padding: 0;
            }
        """)


class TextEdit(QTextEdit):
    def __init__(self, *args):
        super().__init__(*args)
        self.setStyleSheet("""
            QTextEdit {
                border-width: 0;
                border-style: none;
                padding: 0;
            }
        """)
