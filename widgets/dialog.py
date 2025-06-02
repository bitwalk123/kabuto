from PySide6.QtWidgets import QDialog, QDialogButtonBox

from structs.res import AppRes
from widgets.labels import LabelRight, LabelLeft, TextEdit
from widgets.layouts import GridLayout


class DlgAboutThis(QDialog):
    def __init__(self, res: AppRes, progname: str, progver: str):
        super().__init__()
        self.setWindowTitle("このアプリについて")

        layout = GridLayout()
        self.setLayout(layout)

        r = 0
        lab_name_0 = LabelRight("アプリ名")
        layout.addWidget(lab_name_0, r, 0)
        lab_name_1 = LabelLeft(progname)
        layout.addWidget(lab_name_1, r, 1)

        r += 1
        lab_ver_0 = LabelRight("バージョン")
        layout.addWidget(lab_ver_0, r, 0)
        lab_ver_1 = LabelLeft(progver)
        layout.addWidget(lab_ver_1, r, 1)

        r += 1
        lab_author_0 = LabelRight("作　　者")
        layout.addWidget(lab_author_0, r, 0)
        lab_author_1 = LabelLeft("Fuhito Suguri")
        layout.addWidget(lab_author_1, r, 1)

        r += 1
        lab_desc_0 = LabelRight("説　　明")
        layout.addWidget(lab_desc_0, r, 0)
        lab_desc_1 = TextEdit()
        msg = "これはデイトレード用アプリです。\n" \
            "楽天証券の取引ツールであるマーケットスピード２の情報を" \
            "RSS により Excel ワークシート上に読み込み、" \
            "さらに python の xlwings のパッケージが読み込んで処理しています。"
        lab_desc_1.setText(msg)
        layout.addWidget(lab_desc_1, r, 1)

        r += 1
        bbox = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        bbox.accepted.connect(self.accept)
        layout.addWidget(bbox, r, 0, 1, 2)

        layout.setColumnStretch(1, 1)
