from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDockWidget

from widgets.containers import (
    Widget,
)
from widgets.labels import LabelRightMedium
from widgets.layouts import VBoxLayout


class DockWidget(QDockWidget):
    def __init__(self, title: str = ""):
        super().__init__()
        self.title = title

        self.setFeatures(
            QDockWidget.DockWidgetFeature.NoDockWidgetFeatures
        )
        self.lab_title = LabelRightMedium(title)
        self.setTitleBarWidget(self.lab_title)

        base = Widget()
        self.setWidget(base)

        self.layout = layout = VBoxLayout()
        layout.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        layout.setSpacing(2)
        base.setLayout(layout)

    def getTitle(self) -> str:
        """
        タイトル文字列を取得
        :return:
        """
        return self.title

    def setTitle(self, title: str):
        self.lab_title.setText(title)
