from PySide6.QtGui import QFont


class TickFont(QFont):
    def __init__(self):
        super().__init__()
        self.setStyleHint(QFont.StyleHint.Monospace)
        self.setPointSize(9)
