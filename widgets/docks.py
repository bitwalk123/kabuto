from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QDockWidget, QCheckBox, QSizePolicy

from structs.file_path import FilePathModel, FilePathProxyModel, FilePath
from widgets.buttons import (
    Button,
    CheckBoxCrossMA,
    CheckBoxCrossVWAP,
    CheckBoxLosscutVWAP,
    CheckBoxProfitTrailing,
)
from widgets.combos import ComboBox
from widgets.containers import Widget, PadH
from widgets.entries import EntryInt
from widgets.labels import LabelRightMedium, LabelRaisedLeft
from widgets.layouts import VBoxLayout, HBoxLayout, GridLayout
from widgets.listviews import ListViewFileDnD


class DockTitle(Widget):
    def __init__(self, title: str):
        super().__init__()
        layout = HBoxLayout()
        self.setLayout(layout)

        pad = PadH()
        layout.addWidget(pad)

        self.lab_title = LabelRightMedium(title)
        layout.addWidget(self.lab_title)

    def setTitle(self, title: str):
        self.lab_title.setText(title)


class DockWidget(QDockWidget):
    def __init__(self, title: str = ""):
        super().__init__()
        self.title = title

        self.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
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


class DockFileList(QDockWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumWidth(200)
        self.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self.setTitleBarWidget(Widget())

        base = Widget()
        layout = VBoxLayout()
        base.setLayout(layout)
        self.setWidget(base)

        row = HBoxLayout()
        layout.addLayout(row)

        self.chk_sel = chk_sel = QCheckBox("全選択 / 解除")
        chk_sel.setStyleSheet("""
            QCheckBox {
                margin-left: 5px;
                margin-bottom: 2px;
                font-size: 7pt;
            }
        """)
        chk_sel.toggled.connect(self.on_checked)
        row.addWidget(chk_sel)

        # Drag & Drop 用ファイルリスト
        self.model = model = FilePathModel()
        self.lv = lv = ListViewFileDnD(model, self)

        self.proxy = proxy = FilePathProxyModel()
        proxy.setDynamicSortFilter(True)
        proxy.setSourceModel(model)
        proxy.sort(0, Qt.SortOrder.AscendingOrder)

        lv.setModel(proxy)
        layout.addWidget(lv)

    def add_file(self, obj_file: FilePath):
        self.model.add_file(obj_file)

    def on_checked(self, state: bool):
        self.model.set_all_checked(state)

    def get_files(self):
        return self.proxy.files()

    def select_file(self, obj_file: FilePath):
        self.lv.select_file(obj_file)


class DockSimulation(QDockWidget):
    combo_code: ComboBox
    combo_doe: ComboBox

    clickedStart = Signal(dict)

    def __init__(self):
        super().__init__()
        self.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self.setTitleBarWidget(Widget())

        base = Widget()
        layout = VBoxLayout()
        base.setLayout(layout)
        self.setWidget(base)

        layout_code = GridLayout()
        self.gen_row_code(layout_code)
        layout.addLayout(layout_code)

        # クロス MA 返済
        self.cbox_cross_ma = cbox_cross_ma = CheckBoxCrossMA()
        cbox_cross_ma.setChecked(False)
        cbox_cross_ma.stateChanged.connect(self.status_cross_ma_changed)
        layout.addWidget(cbox_cross_ma)

        # クロス VWAP エントリ/返済
        self.cbox_cross_vwap = cbox_cross_vwap = CheckBoxCrossVWAP()
        cbox_cross_vwap.setChecked(True)
        cbox_cross_vwap.stateChanged.connect(self.status_cross_vwap_changed)
        layout.addWidget(cbox_cross_vwap)

        # クロス VWAP 利確
        self.cbox_profit_trailing = cbox_profit_trailing = CheckBoxProfitTrailing()
        cbox_profit_trailing.setChecked(False)
        cbox_profit_trailing.stateChanged.connect(self.status_profit_vwap_changed)
        layout.addWidget(cbox_profit_trailing)

        # クロス VWAP ロスカット
        self.cbox_losscut_vwap = cbox_losscut_vwap = CheckBoxLosscutVWAP()
        cbox_losscut_vwap.setChecked(False)
        cbox_losscut_vwap.stateChanged.connect(self.status_losscut_vwap_changed)
        layout.addWidget(cbox_losscut_vwap)

        but_start = Button("開　始")
        but_start.clicked.connect(self.on_start)
        layout.addWidget(but_start)

        self.on_combo_doe_changed(self.combo_doe.currentText())

    def gen_row_code(self, layout: GridLayout):
        lab_code = LabelRaisedLeft("銘柄コード")
        layout.addWidget(lab_code, 0, 0)

        self.combo_code = combo_code = ComboBox()
        combo_code.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        combo_code.addItems(["9984"])
        layout.addWidget(combo_code, 0, 1)

        lab_doe = LabelRaisedLeft("実験名")
        layout.addWidget(lab_doe, 1, 0)
        self.combo_doe = combo_doe = ComboBox()
        combo_doe.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        combo_doe.addItems([
            "doe-020",
            "doe-019",
            "doe-018",
            "doe-017",
            "doe-016",
            "doe-015",
            "doe-014",
            "doe-013",
            "doe-012",
            "doe-011",
            "doe-010",
            "doe-009",
            "doe-008",
            "doe-007",
            "doe-006",
            "doe-005",
            "doe-004",
            "doe-003",
            "doe-002",
            "doe-001",
        ])
        combo_doe.currentTextChanged.connect(self.on_combo_doe_changed)
        layout.addWidget(combo_doe, 1, 1)

        self.but_condition = but_condition = Button("指定条件")
        but_condition.setCheckable(True)
        layout.addWidget(but_condition, 2, 0)
        self.ent_condition = ent_condition = EntryInt()
        layout.addWidget(ent_condition, 2, 1)


    def on_combo_doe_changed(self, name_doe: str):
        if name_doe == "doe-001":
            self.cbox_cross_ma.setChecked(False)
            self.cbox_cross_vwap.setChecked(True)
            self.cbox_profit_trailing.setChecked(False)
            self.cbox_losscut_vwap.setChecked(False)
        elif name_doe == "doe-002":
            self.cbox_cross_ma.setChecked(False)
            self.cbox_cross_vwap.setChecked(True)
            self.cbox_profit_trailing.setChecked(False)
            self.cbox_losscut_vwap.setChecked(False)
        elif name_doe == "doe-003":
            self.cbox_cross_ma.setChecked(False)
            self.cbox_cross_vwap.setChecked(True)
            self.cbox_profit_trailing.setChecked(True)
            self.cbox_losscut_vwap.setChecked(False)
        elif name_doe == "doe-012":
            self.cbox_cross_ma.setChecked(False)
            self.cbox_cross_vwap.setChecked(True)
            self.cbox_profit_trailing.setChecked(True)
            self.cbox_losscut_vwap.setChecked(False)
        elif name_doe == "doe-013":
            self.cbox_cross_ma.setChecked(False)
            self.cbox_cross_vwap.setChecked(True)
            self.cbox_profit_trailing.setChecked(True)
            self.cbox_losscut_vwap.setChecked(False)
        elif name_doe == "doe-014":
            self.cbox_cross_ma.setChecked(False)
            self.cbox_cross_vwap.setChecked(True)
            self.cbox_profit_trailing.setChecked(True)
            self.cbox_losscut_vwap.setChecked(False)
        elif name_doe == "doe-015":
            self.cbox_cross_ma.setChecked(True)
            self.cbox_cross_vwap.setChecked(False)
            self.cbox_profit_trailing.setChecked(True)
            self.cbox_losscut_vwap.setChecked(False)
        elif name_doe == "doe-016":
            self.cbox_cross_ma.setChecked(True)
            self.cbox_cross_vwap.setChecked(False)
            self.cbox_profit_trailing.setChecked(True)
            self.cbox_losscut_vwap.setChecked(False)
        elif name_doe == "doe-017":
            self.cbox_cross_ma.setChecked(True)
            self.cbox_cross_vwap.setChecked(False)
            self.cbox_profit_trailing.setChecked(False)
            self.cbox_losscut_vwap.setChecked(False)
        elif name_doe == "doe-018":
            self.cbox_cross_ma.setChecked(True)
            self.cbox_cross_vwap.setChecked(False)
            self.cbox_profit_trailing.setChecked(False)
            self.cbox_losscut_vwap.setChecked(False)
        elif name_doe == "doe-019":
            self.cbox_cross_ma.setChecked(True)
            self.cbox_cross_vwap.setChecked(False)
            self.cbox_profit_trailing.setChecked(False)
            self.cbox_losscut_vwap.setChecked(False)
        elif name_doe == "doe-020":
            self.cbox_cross_ma.setChecked(True)
            self.cbox_cross_vwap.setChecked(False)
            self.cbox_profit_trailing.setChecked(False)
            self.cbox_losscut_vwap.setChecked(False)

        else:
            self.cbox_cross_ma.setChecked(False)
            self.cbox_cross_vwap.setChecked(True)
            self.cbox_profit_trailing.setChecked(True)
            self.cbox_losscut_vwap.setChecked(True)

    def on_start(self):
        dict_option = dict()
        # 銘柄コード
        dict_option["code"] = self.combo_code.currentText()
        # 実験番号
        dict_option["doe"] = self.combo_doe.currentText()
        # 指定条件
        if self.but_condition.isChecked():
            num_condition = int(self.ent_condition.text())
            dict_option["condition"] = num_condition
        # 売買条件フラグ
        dict_option["cross_ma"] = self.cbox_cross_ma.isChecked()
        dict_option["cross_vwap"] = self.cbox_cross_vwap.isChecked()
        dict_option["profit_vwap"] = self.cbox_profit_trailing.isChecked()
        dict_option["losscut_vwap"] = self.cbox_losscut_vwap.isChecked()
        self.clickedStart.emit(dict_option)

    def status_cross_ma_changed(self):
        # self.changedStatusCrossMA.emit(self.cbox_cross_ma.isChecked())
        pass

    def status_cross_vwap_changed(self):
        # self.changedStatusCrossVWAP.emit(self.cbox_cross_vwap.isChecked())
        pass

    def status_profit_vwap_changed(self):
        # self.changedStatusProfitVWAP.emit(self.cbox_profit_vwap.isChecked())
        pass

    def status_losscut_vwap_changed(self):
        # self.changedStatusLosscutVWAP.emit(self.cbox_losscut_vwap.isChecked())
        pass
