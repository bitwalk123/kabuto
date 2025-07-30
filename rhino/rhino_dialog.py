import os

from PySide6.QtCore import QMargins, Qt, Signal
from PySide6.QtGui import QPixmap, QIcon
from PySide6.QtWidgets import QDialog, QDialogButtonBox

from structs.res import AppRes
from widgets.buttons import ButtonSmall
from widgets.containers import FrameSunken, PadH
from widgets.entries import EntryRight
from widgets.labels import (
    Label,
    LabelLeft,
    LabelRaised,
    LabelRaisedRight,
    LabelRight,
    PlainTextEdit,
)
from widgets.layouts import GridLayout, HBoxLayout


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
    requestDefaultPSARParams = Signal()
    notifyNewPSARParams = Signal(dict)

    def __init__(self, res: AppRes, code: str, dict_psar: dict):
        super().__init__()
        self.dict_psar = dict_psar

        icon = QIcon(os.path.join(res.dir_image, "setting.png"))
        self.setWindowIcon(icon)
        self.setWindowTitle(f"Setting for {code}")
        self.setStyleSheet("QDialog {font-family: monospace;}")

        layout = GridLayout()
        self.setLayout(layout)

        r = 0
        frame = FrameSunken()
        layout_row = HBoxLayout()
        frame.setLayout(layout_row)
        but_default = ButtonSmall("default")
        but_default.clicked.connect(self.requestDefaultPSARParams.emit)
        layout_row.addWidget(but_default)
        pad = PadH()
        layout_row.addWidget(pad)
        layout.addWidget(frame, r, 0, 1, 2)

        # ---------------------------------------------------------------------
        # Parabolic SAR
        # ---------------------------------------------------------------------
        r += 1
        lab_psar = LabelRaised("Parabolic SAR")
        layout.addWidget(lab_psar, r, 0, 1, 2)

        r += 1
        lab_af_init = LabelRaisedRight("AF (init)")
        layout.addWidget(lab_af_init, r, 0)

        self.ent_af_init = ent_af_init = EntryRight()
        layout.addWidget(ent_af_init, r, 1)

        r += 1
        lab_af_step = LabelRaisedRight("AF (step)")
        layout.addWidget(lab_af_step, r, 0)

        self.ent_af_step = ent_af_step = EntryRight()
        layout.addWidget(ent_af_step, r, 1)

        r += 1
        lab_af_max = LabelRaisedRight("AF (max) ")
        layout.addWidget(lab_af_max, r, 0)

        self.ent_af_max = ent_af_max = EntryRight()
        layout.addWidget(ent_af_max, r, 1)

        r += 1
        lab_factor_d = LabelRaisedRight("Factor D ")
        layout.addWidget(lab_factor_d, r, 0)

        self.ent_factor_d = ent_factor_d = EntryRight()
        layout.addWidget(ent_factor_d, r, 1)

        # ---------------------------------------------------------------------
        # Smoothing
        # ---------------------------------------------------------------------
        r += 1
        lab_psar = LabelRaised("Smoothing")
        layout.addWidget(lab_psar, r, 0, 1, 2)

        r += 1
        lab_power_lam = LabelRaisedRight("power of lam")
        layout.addWidget(lab_power_lam, r, 0)

        self.ent_power_lam = ent_power_lam = EntryRight()
        layout.addWidget(ent_power_lam, r, 1)

        r += 1
        lab_n_smooth_min = LabelRaisedRight("N smooth min")
        layout.addWidget(lab_n_smooth_min, r, 0)

        self.ent_n_smooth_min = ent_n_smooth_min = EntryRight()
        layout.addWidget(ent_n_smooth_min, r, 1)

        r += 1
        lab_n_smooth_max = LabelRaisedRight("N smooth max")
        layout.addWidget(lab_n_smooth_max, r, 0)

        self.ent_n_smooth_max = ent_n_smooth_max = EntryRight()
        layout.addWidget(ent_n_smooth_max, r, 1)

        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # ダイアログ・ボタン
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        r += 1
        bbox = QDialogButtonBox(Qt.Orientation.Horizontal)
        # 「Cancel」ボタン
        bbox.addButton(QDialogButtonBox.StandardButton.Cancel)
        bbox.rejected.connect(self.button_cancel_clicked)
        # 「Ok」ボタン
        bbox.addButton(QDialogButtonBox.StandardButton.Ok)
        bbox.accepted.connect(self.button_ok_clicked)
        layout.addWidget(bbox, r, 0, 1, 2)

        layout.setColumnStretch(1, 1)

        # 辞書の内容を表示に転記
        self.set_psar_params(dict_psar)

    def button_cancel_clicked(self):
        self.reject()

    def button_ok_clicked(self):
        dict_psar = self.get_entries()
        print(dict_psar)
        self.accept()

    def set_default_psar_params(self, dict_default_psar: dict):
        self.set_psar_params(dict_default_psar)

    def get_entries(self) -> dict:
        dict_psar = dict()

        # ---------------------------------------------------------------------
        # Parabolic SAR
        # ---------------------------------------------------------------------
        dict_psar['af_init'] = float(self.ent_af_init.text())
        dict_psar['af_step'] = float(self.ent_af_step.text())
        dict_psar['af_max'] = float(self.ent_af_max.text())
        dict_psar['factor_d'] = float(self.ent_factor_d.text())
        # ---------------------------------------------------------------------
        # Smoothing
        # ---------------------------------------------------------------------
        dict_psar['power_lam'] = int(self.ent_power_lam.text())
        dict_psar['n_smooth_min'] = int(self.ent_n_smooth_min.text())
        dict_psar['n_smooth_max'] = int(self.ent_n_smooth_max.text())

        return dict_psar

    def set_psar_params(self, dict_psar: dict):
        """
        辞書の内容を表示に転記
        :param dict_psar:
        :return:
        """
        # ---------------------------------------------------------------------
        # Parabolic SAR
        # ---------------------------------------------------------------------
        self.ent_af_init.setText(f"{dict_psar['af_init']:f}")
        self.ent_af_step.setText(f"{dict_psar['af_step']:f}")
        self.ent_af_max.setText(f"{dict_psar['af_max']:f}")
        self.ent_factor_d.setText(f"{dict_psar['factor_d']:f}")
        # ---------------------------------------------------------------------
        # Smoothing
        # ---------------------------------------------------------------------
        self.ent_power_lam.setText(f"{dict_psar['power_lam']:d}")
        self.ent_n_smooth_min.setText(f"{dict_psar['n_smooth_min']:d}")
        self.ent_n_smooth_max.setText(f"{dict_psar['n_smooth_max']:d}")
