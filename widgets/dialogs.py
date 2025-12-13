import os

from PySide6.QtCore import QMargins, Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import QDialog, QDialogButtonBox

from structs.res import AppRes
from widgets.entries import EntryFloat
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
        self.setWindowTitle("このアプリについて")
        self.setStyleSheet("""
            QDialog {
                font-family: monospace;
            }
        """)

        layout = GridLayout()
        self.setLayout(layout)

        r = 0
        lab_name_0 = LabelRight("アプリ名")
        layout.addWidget(lab_name_0, r, 0)
        lab_name_1 = LabelLeft(progname)
        layout.addWidget(lab_name_1, r, 1)
        lab_name_2 = Label()
        pixmap = QPixmap(os.path.join(res.dir_image, "kabuto.png")).scaledToWidth(64)
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
        lab_desc_0 = LabelRight("説　　明")
        layout.addWidget(lab_desc_0, r, 0)
        lab_desc_1 = PlainTextEdit()
        msg = "これはデイトレード用アプリです。\n" \
              "楽天証券の取引ツールである「マーケットスピード２」の情報を" \
              "RSS により Excel ワークシート上に読み込み、" \
              "さらに python の xlwings のパッケージを利用して" \
              "ワークシート上を読み書きして処理しています。"
        lab_desc_1.setPlainText(msg)
        lab_desc_1.setReadOnly(True)
        layout.addWidget(lab_desc_1, r, 1, 1, 2)

        r += 1
        bbox = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        bbox.accepted.connect(self.accept)
        layout.addWidget(bbox, r, 0, 1, 3)

        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)


class DlgAboutThis2(QDialog):
    def __init__(
            self,
            res: AppRes,
            progname: str,
            progver: str,
            author: str,
            license: str,
            name_icon: str = "kabuto.png",
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
        pixmap = QPixmap(os.path.join(res.dir_image, name_icon)).scaledToWidth(64)
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


class DlgParam(QDialog):
    def __init__(self, res: AppRes, code: str, dict_setting: dict):
        super().__init__()
        self.res = res
        self.code = code
        self.dict_setting = dict_setting

        icon = QIcon(os.path.join(res.dir_image, "setting.png"))
        self.setWindowIcon(icon)
        self.setWindowTitle(f"パラメータ ({code})")

        self.setStyleSheet("QDialog {font-family: monospace;}")

        layout = GridLayout()
        self.setLayout(layout)

        r = 0
        lab_head_0 = LabelRaised("パラメータ")
        layout.addWidget(lab_head_0, r, 0)

        lab_head_1 = LabelRaised("設定値")
        layout.addWidget(lab_head_1, r, 1)

        r += 1
        param = "PERIOD_MA_1"
        lab_param_1 = LabelRaisedLeft(param)
        layout.addWidget(lab_param_1, r, 0)

        value_str = str(dict_setting[param])
        self.obj_period_ma_1 = ent_param_1 = EntryFloat(value_str)
        layout.addWidget(ent_param_1, r, 1)

        r += 1
        param = "PERIOD_MA_2"
        lab_param_2 = LabelRaisedLeft(param)
        layout.addWidget(lab_param_2, r, 0)

        value_str = str(dict_setting[param])
        self.obj_period_ma_2 = ent_param_2 = EntryFloat(value_str)
        layout.addWidget(ent_param_2, r, 1)

        r += 1
        param = "PERIOD_MR"
        lab_param_3 = LabelRaisedLeft(param)
        layout.addWidget(lab_param_3, r, 0)

        value_str = str(dict_setting[param])
        self.obj_period_mr = ent_param_3 = EntryFloat(value_str)
        layout.addWidget(ent_param_3, r, 1)

        r += 1
        param = "THRESHOLD_MR"
        lab_param_4 = LabelRaisedLeft(param)
        layout.addWidget(lab_param_4, r, 0)

        value_str = str(dict_setting[param])
        self.obj_threshold_mr = ent_param_4 = EntryFloat(value_str)
        layout.addWidget(ent_param_4, r, 1)

        r += 1
        bbox = QDialogButtonBox()
        bbox.addButton(QDialogButtonBox.StandardButton.Ok)
        bbox.addButton(QDialogButtonBox.StandardButton.Cancel)
        bbox.accepted.connect(self.accept)
        layout.addWidget(bbox, r, 0, 1, 2)
