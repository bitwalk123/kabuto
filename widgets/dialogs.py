import os

from PySide6.QtCore import QMargins, Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
)

from structs.res import AppRes
from widgets.entries import EntryFloat, EntryInt
from widgets.labels import (
    Label,
    LabelLeft,
    LabelRaised,
    LabelRaisedLeft,
    LabelRight,
    PlainTextEdit,
)
from widgets.layouts import GridLayout, VBoxLayout
from widgets.listviews import CheckList


class DlgAboutThis(QDialog):
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
        self.setContentsMargins(QMargins(5, 5, 5, 5))
        self.setStyleSheet("QDialog {font-family: monospace;}")

        icon = QIcon(os.path.join(res.dir_image, "about.png"))
        self.setWindowIcon(icon)
        self.setWindowTitle("このアプリについて")

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
        msg = (
            "これはデイトレード用アプリです。\n"
            "Python の xlwings パッケージを利用して、"
            "Excel シート (MARKET SPEED II RSS) とやりとりをします。\n"
            "MARKET SPEED II RSS は、Microsoftの表計算ソフト Excel の"
            "アドインとして利用できるトレーディングツールです。"
        )
        lab_desc.setPlainText(msg)
        lab_desc.setReadOnly(True)
        layout.addWidget(lab_desc, r, 0, 1, 3)

        r += 1
        bbox = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        bbox.accepted.connect(self.accept)
        layout.addWidget(bbox, r, 0, 1, 3)

        layout.setColumnStretch(1, 1)

    def showEvent(self, event):
        super().showEvent(event)
        # 表示後の最終サイズを固定
        self.setFixedSize(self.size())


class DlgCodeSel(QDialog):
    def __init__(self, list_code: list, row_default: int = 0):
        super().__init__()
        self.setWindowTitle("銘柄コード一覧")

        layout = VBoxLayout()
        self.setLayout(layout)

        self.clist = clist = CheckList()
        clist.addItems(list_code, row_default)
        layout.addWidget(clist)

        bbox = QDialogButtonBox()
        bbox.addButton(QDialogButtonBox.StandardButton.Ok)
        bbox.addButton(QDialogButtonBox.StandardButton.Cancel)
        bbox.accepted.connect(self.accept)
        bbox.rejected.connect(self.reject)
        layout.addWidget(bbox)

    def getSelected(self) -> list:
        return self.clist.getSelected()


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

        value_str = str(dict_setting.get(param, 60))
        self.obj_period_ma_1 = ent_param_1 = EntryInt(value_str)
        layout.addWidget(ent_param_1, r, 1)

        r += 1
        param = "PERIOD_MA_2"
        lab_param_2 = LabelRaisedLeft(param)
        layout.addWidget(lab_param_2, r, 0)

        value_str = str(dict_setting.get(param, 600))
        self.obj_period_ma_2 = ent_param_2 = EntryInt(value_str)
        layout.addWidget(ent_param_2, r, 1)

        r += 1
        param = "PERIOD_MR"
        lab_param_3 = LabelRaisedLeft(param)
        layout.addWidget(lab_param_3, r, 0)

        value_str = str(dict_setting.get(param, 30))
        self.obj_period_mr = ent_param_3 = EntryInt(value_str)
        layout.addWidget(ent_param_3, r, 1)

        r += 1
        param = "THRESHOLD_MR"
        lab_param_4 = LabelRaisedLeft(param)
        layout.addWidget(lab_param_4, r, 0)

        value_str = str(dict_setting.get(param, 7))
        self.obj_threshold_mr = ent_param_4 = EntryFloat(value_str)
        layout.addWidget(ent_param_4, r, 1)

        r += 1
        param = "LOSSCUT_1"
        lab_param_5 = LabelRaisedLeft(param)
        layout.addWidget(lab_param_5, r, 0)

        value_str = str(dict_setting.get(param, -1.0e8))
        self.obj_losscut_1 = ent_param_5 = EntryFloat(value_str)
        layout.addWidget(ent_param_5, r, 1)

        r += 1
        bbox = QDialogButtonBox()
        bbox.addButton(QDialogButtonBox.StandardButton.Ok)
        bbox.addButton(QDialogButtonBox.StandardButton.Cancel)
        bbox.accepted.connect(self.accept)
        layout.addWidget(bbox, r, 0, 1, 2)

    def getParam(self) -> dict:
        dict_param = dict()
        dict_param["PERIOD_MA_1"] = self.obj_period_ma_1.getValue()
        dict_param["PERIOD_MA_2"] = self.obj_period_ma_2.getValue()
        dict_param["PERIOD_MR"] = self.obj_period_mr.getValue()
        dict_param["THRESHOLD_MR"] = self.obj_threshold_mr.getValue()
        dict_param["LOSSCUT_1"] = self.obj_losscut_1.getValue()
        return dict_param

    def showEvent(self, event):
        super().showEvent(event)
        # 表示後の最終サイズを固定
        self.setFixedSize(self.size())


class DlgTickFileSel(QFileDialog):
    def __init__(self, res: AppRes):
        super().__init__()
        self.setWindowIcon(QIcon(os.path.join(res.dir_image, "excel.png")))
        self.setOption(QFileDialog.Option.DontUseNativeDialog)
        self.setDefaultSuffix("xlsx")
        self.setDirectory(res.dir_collection)
