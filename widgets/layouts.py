from PySide6.QtCore import QMargins
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QGridLayout


class HBoxLayout(QHBoxLayout):
    def __init__(self):
        super().__init__()
        self.setContentsMargins(QMargins(0, 0, 0, 0))
        self.setSpacing(0)


class VBoxLayout(QVBoxLayout):
    def __init__(self):
        super().__init__()
        self.setContentsMargins(QMargins(0, 0, 0, 0))
        self.setSpacing(0)


class GridLayout(QGridLayout):
    def __init__(self):
        super().__init__()
        self.setContentsMargins(QMargins(0, 0, 0, 0))
        self.setSpacing(0)