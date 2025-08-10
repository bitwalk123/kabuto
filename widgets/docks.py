from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDockWidget

from widgets.containers import Widget, PadH
from widgets.labels import LabelRightMedium
from widgets.layouts import VBoxLayout, HBoxLayout
from widgets.switches import Switch


class DockTitle(Widget):
    def __init__(self, title: str):
        super().__init__()
        layout = HBoxLayout()
        self.setLayout(layout)

        pad = PadH()
        layout.addWidget(pad)

        self.lab_title = LabelRightMedium(title)
        layout.addWidget(self.lab_title)

        self.switch = switch = Switch()
        switch.set(False)
        switch.setToolTip("RSS売買 ON/OFF")
        switch.statusChanged.connect(self.changed_swicth_status)
        layout.addWidget(switch)

    def changed_swicth_status(self, state: bool):
        print(state)

    def isSwitchChecked(self):
        self.switch.isChecked()

    def setTitle(self, title: str):
        self.lab_title.setText(title)


class DockWidget(QDockWidget):
    def __init__(self, title: str = ""):
        super().__init__()
        self.title = title

        self.setFeatures(
            QDockWidget.DockWidgetFeature.NoDockWidgetFeatures
        )
        self.dock_title = DockTitle(title)
        self.setTitleBarWidget(self.dock_title)

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
        self.dock_title.setTitle(title)
