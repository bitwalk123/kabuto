import os

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QDialog, QDialogButtonBox

from structs.res import AppRes
from widgets.buttons import ButtonSmall
from widgets.containers import FrameSunken, PadH
from widgets.entries import EntryRight
from widgets.labels import (
    LabelRaised,
    LabelRaisedRight,
)
from widgets.layouts import GridLayout, HBoxLayout


class DlgTradeConfigPSAR(QDialog):
    requestDefaultPSARParams = Signal()
    notifyNewPSARParams = Signal(dict)

    def __init__(self, res: AppRes, code: str, dict_psar: dict):
        super().__init__()
        self.dict_psar = dict_psar
        self.dict_entry = dict()

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
        but_default.clicked.connect(self.request_default_psar_params)
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

        self.dict_entry["af_init"] = ent_af_init = EntryRight()
        layout.addWidget(ent_af_init, r, 1)

        r += 1
        lab_af_step = LabelRaisedRight("AF (step)")
        layout.addWidget(lab_af_step, r, 0)

        self.dict_entry["af_step"] = ent_af_step = EntryRight()
        layout.addWidget(ent_af_step, r, 1)

        r += 1
        lab_af_max = LabelRaisedRight("AF (max) ")
        layout.addWidget(lab_af_max, r, 0)

        self.dict_entry["af_max"] = ent_af_max = EntryRight()
        layout.addWidget(ent_af_max, r, 1)

        r += 1
        lab_factor_d = LabelRaisedRight("Factor D ")
        layout.addWidget(lab_factor_d, r, 0)

        self.dict_entry["factor_d"] = ent_factor_d = EntryRight()
        layout.addWidget(ent_factor_d, r, 1)

        r += 1
        lab_factor_c = LabelRaisedRight("Factor C ")
        layout.addWidget(lab_factor_c, r, 0)

        self.dict_entry["factor_c"] = ent_factor_c = EntryRight()
        layout.addWidget(ent_factor_c, r, 1)

        # ---------------------------------------------------------------------
        # Smoothing
        # ---------------------------------------------------------------------
        r += 1
        lab_psar = LabelRaised("Smoothing")
        layout.addWidget(lab_psar, r, 0, 1, 2)

        r += 1
        lab_power_lam = LabelRaisedRight("power of lam")
        layout.addWidget(lab_power_lam, r, 0)

        self.dict_entry["power_lam"] = ent_power_lam = EntryRight()
        layout.addWidget(ent_power_lam, r, 1)

        r += 1
        lab_n_smooth_min = LabelRaisedRight("N smooth min")
        layout.addWidget(lab_n_smooth_min, r, 0)

        self.dict_entry["n_smooth_min"] = ent_n_smooth_min = EntryRight()
        layout.addWidget(ent_n_smooth_min, r, 1)

        r += 1
        lab_n_smooth_max = LabelRaisedRight("N smooth max")
        layout.addWidget(lab_n_smooth_max, r, 0)

        self.dict_entry["n_smooth_max"] = ent_n_smooth_max = EntryRight()
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
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 Parabolic SAR 関連の新しいパラメータを通知
        self.notifyNewPSARParams.emit(dict_psar)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.accept()

    def get_entries(self) -> dict:
        dict_psar = dict()

        # ---------------------------------------------------------------------
        # Parabolic SAR
        # ---------------------------------------------------------------------
        for key in ["af_init", "af_step", "af_max", "factor_d", "factor_c"]:
            dict_psar[key] = float(self.dict_entry[key].text())
        # ---------------------------------------------------------------------
        # Smoothing
        # ---------------------------------------------------------------------
        for key in ["power_lam", "n_smooth_min", "n_smooth_max"]:
            dict_psar[key] = int(self.dict_entry[key].text())

        return dict_psar

    def request_default_psar_params(self):
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 Parabolic SAR 関連のパラメータを要求
        self.requestDefaultPSARParams.emit()
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def set_default_psar_params(self, dict_default_psar: dict):
        self.set_psar_params(dict_default_psar)

    def set_psar_params(self, dict_psar: dict):
        """
        辞書の内容を表示に転記
        :param dict_psar:
        :return:
        """
        # ---------------------------------------------------------------------
        # Parabolic SAR
        # ---------------------------------------------------------------------
        for key in ["af_init", "af_step", "af_max", "factor_d", "factor_c"]:
            entry: EntryRight = self.dict_entry[key]
            entry.setText(f"{dict_psar[key]:f}")
        # ---------------------------------------------------------------------
        # Smoothing
        # ---------------------------------------------------------------------
        for key in ["power_lam", "n_smooth_min", "n_smooth_max"]:
            entry: EntryRight = self.dict_entry[key]
            entry.setText(f"{dict_psar[key]:d}")
