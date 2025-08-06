from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame

from rhino.rhino_dialog import DlgTradeConfig
from structs.res import AppRes
from widgets.buttons import (
    ButtonSave,
    ButtonSetting,
    ToggleButtonAutoPilot,
    TradeButton, ToggleButtonOverDrive,
)
from widgets.containers import (
    IndicatorBuySell,
    PadH,
    Widget,
)
from widgets.layouts import GridLayout, HBoxLayout


class PanelTrading(Widget):
    """
    トレーディング用パネル
    固定株数でナンピンしない取引を前提にしている
    """
    clickedBuy = Signal()
    clickedSell = Signal()
    clickedRepay = Signal()

    def __init__(self):
        super().__init__()
        layout = GridLayout()
        self.setLayout(layout)

        row = 0
        # 建玉の売建（インジケータ）
        self.ind_sell = ind_sell = IndicatorBuySell()
        layout.addWidget(ind_sell, row, 0)

        # 建玉の買建（インジケータ）
        self.ind_buy = ind_buy = IndicatorBuySell()
        layout.addWidget(ind_buy, row, 1)

        row += 1
        # 建玉の売建
        self.sell = but_sell = TradeButton("sell")
        but_sell.clicked.connect(self.on_sell)
        layout.addWidget(but_sell, row, 0)

        # 建玉の買建
        self.buy = but_buy = TradeButton("buy")
        but_buy.clicked.connect(self.on_buy)
        layout.addWidget(but_buy, row, 1)

        row += 1
        # 建玉の返却
        self.repay = but_repay = TradeButton("repay")
        but_repay.clicked.connect(self.on_repay)
        layout.addWidget(but_repay, row, 0, 1, 2)

        # 初期状態ではポジション無し
        self.position_close()

    def position_close(self):
        self.sell.setEnabled(True)
        self.buy.setEnabled(True)
        self.repay.setDisabled(True)

    def position_open(self):
        self.sell.setDisabled(True)
        self.buy.setDisabled(True)
        self.repay.setEnabled(True)

    def on_buy(self):
        # ---------------------------------------------------------------------
        # 🧿 買建ボタンがクリックされたことを通知
        self.clickedBuy.emit()
        # ---------------------------------------------------------------------
        self.position_open()
        self.ind_buy.setBuy()

    def on_sell(self):
        # ---------------------------------------------------------------------
        # 🧿 売建ボタンがクリックされたことを通知
        self.clickedSell.emit()
        # ---------------------------------------------------------------------
        self.position_open()
        self.ind_sell.setSell()

    def on_repay(self):
        # ---------------------------------------------------------------------
        # 🧿 返却ボタンがクリックされたことを通知
        self.clickedRepay.emit()
        # ---------------------------------------------------------------------
        self.position_close()
        self.ind_buy.setDefault()
        self.ind_sell.setDefault()


class PanelOption(QFrame):
    """
    トレーディング用オプションパネル
    """
    requestDefaultPSARParams = Signal()
    requestPSARParams = Signal()
    notifyNewPSARParams = Signal(dict)

    def __init__(self, res: AppRes, code: str):
        super().__init__()
        self.res = res
        self.code = code
        self.dlg: DlgTradeConfig | None = None

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
        layout.addWidget(but_overdrive)

    def isAutoPilotEnabled(self) -> bool:
        return self.autopilot.isChecked()

    def notify_new_psar_params(self, dict_psar: dict):
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 Parabolic SAR 関連の新しいパラメータを通知
        self.notifyNewPSARParams.emit(dict_psar)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def request_default_psar_params(self):
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 Parabolic SAR 関連のデフォルトのパラメータを要求
        self.requestDefaultPSARParams.emit()
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def set_default_psar_params(self, dict_default_psar: dict):
        if self.dlg is not None:
            self.dlg.set_default_psar_params(dict_default_psar)

    def setAutoPilotEnabled(self, state: bool = True):
        self.autopilot.setChecked(state)

    def trade_config(self):
        self.requestPSARParams.emit()

    def show_trade_config(self, dict_psar: dict):
        self.dlg = dlg = DlgTradeConfig(self.res, self.code, dict_psar)
        dlg.requestDefaultPSARParams.connect(self.request_default_psar_params)
        dlg.notifyNewPSARParams.connect(self.notify_new_psar_params)
        dlg.exec()
