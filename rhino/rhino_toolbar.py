from PySide6.QtWidgets import QToolBar

from structs.res import AppRes


class RhinoToolBar(QToolBar):
    def __init__(self, res: AppRes):
        super().__init__()
        self.res = res
