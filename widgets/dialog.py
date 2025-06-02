from PySide6.QtWidgets import QDialog, QDialogButtonBox

from structs.res import AppRes
from widgets.labels import LabelRight, LabelLeft
from widgets.layouts import GridLayout


class DlgAboutThis(QDialog):
    def __init__(self, res: AppRes, progname: str, progver: str):
        super().__init__()
        self.setWindowTitle('このアプリについて')

        layout = GridLayout()
        self.setLayout(layout)

        r = 0
        lab_name_0 = LabelRight("アプリ名 ")
        layout.addWidget(lab_name_0, r, 0)
        lab_name_1 = LabelLeft(progname)
        layout.addWidget(lab_name_1, r, 1)

        r += 1
        lab_ver_0 = LabelRight("バージョン ")
        layout.addWidget(lab_ver_0, r, 0)
        lab_ver_1 = LabelLeft(progver)
        layout.addWidget(lab_ver_1, r, 1)

        bbox = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        bbox.accepted.connect(self.accept)
        layout.addWidget(bbox)
