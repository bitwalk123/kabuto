import os

from PySide6.QtCore import QMargins, Qt
from PySide6.QtGui import QPixmap, QIcon
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QPushButton

from structs.res import AppRes
from widgets.entries import Entry
from widgets.labels import (
    Label,
    LabelLeft,
    LabelRaised,
    LabelRaisedLeft,
    LabelRight,
    PlainTextEdit,
)
from widgets.layouts import GridLayout


class DlgAboutThis(QDialog):
    def __init__(
            self,
            res: AppRes,
            progname: str,
            progver: str,
            author: str,
            license: str
    ):
        super().__init__()

        icon = QIcon(os.path.join(res.dir_image, "about.png"))
        self.setWindowIcon(icon)
        self.setWindowTitle("このアプリについて")
        self.setStyleSheet("QDialog {font-family: monospace;}")

        layout = GridLayout()
        self.setLayout(layout)

        r = 0
        lab_name_0 = LabelRight("アプリ名")
        layout.addWidget(lab_name_0, r, 0)
        lab_name_1 = LabelLeft(progname)
        layout.addWidget(lab_name_1, r, 1)
        lab_name_2 = Label()
        pixmap = QPixmap(os.path.join(res.dir_image, "rhino.png")).scaledToWidth(64)
        lab_name_2.setPixmap(pixmap)
        lab_name_2.setContentsMargins(QMargins(5, 0, 5, 0))
        lab_name_2.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(lab_name_2, r, 2, 4, 1)

        r += 1
        lab_ver_0 = LabelRight("バージョン")
        layout.addWidget(lab_ver_0, r, 0)
        lab_ver_1 = LabelLeft(progver)
        layout.addWidget(lab_ver_1, r, 1)

        r += 1
        lab_author_0 = LabelRight("作　　者")
        layout.addWidget(lab_author_0, r, 0)
        lab_author_1 = LabelLeft(author)
        layout.addWidget(lab_author_1, r, 1)

        r += 1
        lab_license_0 = LabelRight("ライセンス")
        layout.addWidget(lab_license_0, r, 0)
        lab_license_1 = LabelLeft(license)
        layout.addWidget(lab_license_1, r, 1)

        r += 1
        lab_desc = PlainTextEdit()
        msg = "これはデイトレード用アプリです。\n" \
              "楽天証券が提供している取引ツール MARKET SPEED II RSS に対して、" \
              "Python の xlwings のパッケージを利用してやりとりをします。"
        lab_desc.setPlainText(msg)
        lab_desc.setReadOnly(True)
        layout.addWidget(lab_desc, r, 0, 1, 3)

        r += 1
        bbox = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        bbox.accepted.connect(self.accept)
        layout.addWidget(bbox, r, 0, 1, 3)

        layout.setColumnStretch(1, 1)


class DlgTradeConfig(QDialog):
    def __init__(self, res: AppRes, code: str):
        super().__init__()

        icon = QIcon(os.path.join(res.dir_image, "setting.png"))
        self.setWindowIcon(icon)
        self.setWindowTitle(f"Setting for {code}")
        self.setStyleSheet("QDialog {font-family: monospace;}")

        layout = GridLayout()
        self.setLayout(layout)

        r = 0
        lab_psar = LabelRaised("Parabolic SAR")
        layout.addWidget(lab_psar, r, 0, 1, 2)

        r += 1
        lab_af_init = LabelRaisedLeft("AF (init) ")
        layout.addWidget(lab_af_init, r, 0)
        ent_af_init = Entry()
        layout.addWidget(ent_af_init, r, 1)

        r += 1
        lab_af_step = LabelRaisedLeft("AF (step) ")
        layout.addWidget(lab_af_step, r, 0)
        ent_af_step = Entry()
        layout.addWidget(ent_af_step, r, 1)

        r += 1
        lab_af_max = LabelRaisedLeft("AF (max)")
        layout.addWidget(lab_af_max, r, 0)
        ent_af_max = Entry()
        layout.addWidget(ent_af_max, r, 1)

        r += 1
        bbox = QDialogButtonBox(Qt.Orientation.Horizontal)
        # 「Cancel」ボタン
        bbox.addButton(QDialogButtonBox.StandardButton.Cancel)
        bbox.rejected.connect(self.reject)
        # 「Ok」ボタン
        bbox.addButton(QDialogButtonBox.StandardButton.Ok)
        bbox.accepted.connect(self.accept)
        layout.addWidget(bbox, r, 0, 1, 2)

        layout.setColumnStretch(1, 1)
