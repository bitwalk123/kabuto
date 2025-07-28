from PySide6.QtWidgets import QStatusBar

from structs.res import AppRes


class RhinoStatusBar(QStatusBar):
    def __init__(self, res: AppRes):
        super().__init__()
        self.res = res
