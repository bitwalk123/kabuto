from PySide6.QtCore import QMargins, Qt
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QGridLayout


class HBoxLayout(QHBoxLayout):
    def __init__(self):
        super().__init__()
        self.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        self.setContentsMargins(QMargins(0, 0, 0, 0))
        self.setSpacing(0)


class VBoxLayout(QVBoxLayout):
    def __init__(self):
        super().__init__()
        self.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        self.setContentsMargins(QMargins(0, 0, 0, 0))
        self.setSpacing(0)



class GridLayout(QGridLayout):
    def __init__(self):
        super().__init__()
        self.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        self.setContentsMargins(QMargins(0, 0, 0, 0))
        self.setSpacing(0)