import os

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QDialog, QDialogButtonBox

from structs.res import AppRes
from widgets.containers import TabWidget, Widget
from widgets.dialogs import DialogButtonBox
from widgets.layouts import VBoxLayout


class DlgSystemSettings(QDialog):
    def __init__(self, res: AppRes):
        super().__init__()
        self.res = res

        icon = QIcon(os.path.join(res.dir_image, "setting.png"))
        self.setWindowIcon(icon)
        self.setWindowTitle("システム設定")

        self.setStyleSheet("QDialog {font-family: %s;}" % res.name_tick_font)

        layout = VBoxLayout()
        self.setLayout(layout)

        base = TabWidget()
        layout.addWidget(base)

        base.addTab(Widget(), "メイン")


        bbox = DialogButtonBox()
        bbox.addButton(DialogButtonBox.StandardButton.Ok)
        bbox.addButton(DialogButtonBox.StandardButton.Cancel)
        bbox.accepted.connect(self.accept)
        bbox.rejected.connect(self.reject)
        layout.addWidget(bbox)
