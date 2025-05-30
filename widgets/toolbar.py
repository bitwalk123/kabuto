from PySide6.QtWidgets import QToolBar, QToolButton

from structs.res import AppRes


class ToolBar(QToolBar):
    def __init__(self, res: AppRes):
        super().__init__()
        self.res = res

        if res.debug:
            but_open = QToolButton()
            icon_open = res.getBuiltinIcon(self, "DialogOpenButton")
            but_open.setIcon(icon_open)
            self.addWidget(but_open)

        but_info = QToolButton()
        icon_info = res.getBuiltinIcon(self, "MessageBoxInformation")
        but_info.setIcon(icon_info)
        self.addWidget(but_info)
