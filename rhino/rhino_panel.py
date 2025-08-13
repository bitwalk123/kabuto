from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame

from rhino.rhino_dialog import DlgTradeConfigPSAR
from structs.res import AppRes
from widgets.buttons import (
    ButtonSave,
    ButtonSetting,
    ToggleButtonAutoPilot,
    ToggleButtonOverDrive,
)
from widgets.containers import PadH
from widgets.layouts import HBoxLayout


class PanelOption4PSAR(QFrame):
    """
    トレーディング用オプションパネル
    """
    requestDefaultPSARParams = Signal()
    requestPSARParams = Signal()
    requestOEStatusChange = Signal(bool)
    notifyNewPSARParams = Signal(dict)

    def __init__(self, res: AppRes, code: str):
        super().__init__()
        self.res = res
        self.code = code
        self.dlg: DlgTradeConfigPSAR | None = None

        self.setFrameStyle(
            QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken
        )
        self.setLineWidth(1)
        layout = HBoxLayout()
        self.setLayout(layout)

        self.autopilot = but_autopilot = ToggleButtonAutoPilot(res)
        but_autopilot.setChecked(True)  # デフォルトで ON
        layout.addWidget(but_autopilot)

        pad = PadH()
        layout.addWidget(pad)

        self.save = but_save = ButtonSave(res)
        layout.addWidget(but_save)

        self.setting = but_setting = ButtonSetting(res)
        but_setting.clicked.connect(self.trade_config)
        layout.addWidget(but_setting)

        self.overdrive = but_overdrive = ToggleButtonOverDrive(res)
        # クリック操作を区別するために toggled を使わずに clicked シグナルを使う
        but_overdrive.clicked.connect(self.over_drive_clicked)
        layout.addWidget(but_overdrive)

    def isAutoPilotEnabled(self) -> bool:
        return self.autopilot.isChecked()

    def isOverDriveEnabled(self) -> bool:
        return self.overdrive.isChecked()

    def notify_new_psar_params(self, dict_psar: dict):
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 Parabolic SAR 関連の新しいパラメータを通知
        self.notifyNewPSARParams.emit(dict_psar)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def over_drive_clicked(self):
        """
        クリック操作を区別するために toggled シグナルを使わずに clicked を使う
        :return:
        """
        self.requestOEStatusChange.emit(self.overdrive.isChecked())

    def request_default_psar_params(self):
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 Parabolic SAR 関連のデフォルトのパラメータを要求
        self.requestDefaultPSARParams.emit()
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def setDefaultPSARParams(self, dict_default_psar: dict):
        if self.dlg is not None:
            self.dlg.set_default_psar_params(dict_default_psar)

    def setAutoPilotEnabled(self, state: bool = True):
        self.autopilot.setChecked(state)

    def setOverDriveEnabled(self, state: bool = True):
        self.overdrive.setChecked(state)

    def trade_config(self):
        self.requestPSARParams.emit()

    def showTradeConfig(self, dict_psar: dict):
        self.dlg = dlg = DlgTradeConfigPSAR(self.res, self.code, dict_psar)
        dlg.requestDefaultPSARParams.connect(self.request_default_psar_params)
        dlg.notifyNewPSARParams.connect(self.notify_new_psar_params)
        dlg.exec()
