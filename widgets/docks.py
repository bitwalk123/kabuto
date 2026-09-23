from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QDockWidget, QCheckBox, QPushButton

from structs.file_path import FilePathModel, FilePathProxyModel, FilePath
from widgets.buttons import (
    CheckBoxCrossMA,
    CheckBoxCrossVWAP,
    CheckBoxLosscutVWAP,
    CheckBoxProfitVWAP,
)
from widgets.containers import Widget, PadH
from widgets.labels import LabelRightMedium
from widgets.layouts import VBoxLayout, HBoxLayout
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
    clickedStart = Signal(dict)

    def __init__(self):
        super().__init__()

        base = Widget()
        layout = VBoxLayout()
        base.setLayout(layout)
        self.setWidget(base)

        # クロス MA 返済
        self.cbox_cross_ma = cbox_cross_ma = CheckBoxCrossMA()
        cbox_cross_ma.setCheckable(False)
        cbox_cross_ma.stateChanged.connect(self.status_cross_ma_changed)
        layout.addWidget(cbox_cross_ma)

        # クロス VWAP エントリ/返済
        self.cbox_cross_vwap = cbox_cross_vwap = CheckBoxCrossVWAP()
        cbox_cross_vwap.setChecked(True)
        cbox_cross_vwap.stateChanged.connect(self.status_cross_vwap_changed)
        layout.addWidget(cbox_cross_vwap)

        # クロス VWAP 利確
        self.cbox_profit_vwap = cbox_profit_vwap = CheckBoxProfitVWAP()
        cbox_profit_vwap.setChecked(True)
        cbox_profit_vwap.stateChanged.connect(self.status_profit_vwap_changed)
        layout.addWidget(cbox_profit_vwap)

        # クロス VWAP ロスカット
        self.cbox_losscut_vwap = cbox_losscut_vwap = CheckBoxLosscutVWAP()
        cbox_losscut_vwap.setChecked(True)
        cbox_losscut_vwap.stateChanged.connect(self.status_losscut_vwap_changed)
        layout.addWidget(cbox_losscut_vwap)

        but_start = QPushButton("開　始")
        but_start.clicked.connect(self.on_start)
        layout.addWidget(but_start)

    def on_start(self):
        dict_option = dict()
        dict_option["cross_ma"] = self.cbox_cross_ma.isChecked()
        dict_option["cross_vwap"] = self.cbox_cross_vwap.isChecked()
        dict_option["profit_vwap"] = self.cbox_profit_vwap.isChecked()
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
