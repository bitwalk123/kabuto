from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame

from structs.res import AppRes
from widgets.buttons import (
    ButtonSave,
    ButtonSetting,
    TradeButton,
    ToggleButtonAutoPilot,
)
from widgets.containers import (
    IndicatorBuySell,
    Widget, PadH,
)
from widgets.layouts import (
    GridLayout,
    HBoxLayout,
)


class PanelTrading(Widget):
    """
    トレーディング用パネル
    固定株数でナンピンしない取引を前提にしている
    """
    clickedBuy = Signal()
    clickedRepay = Signal()
    clickedSell = Signal()

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
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 買建ボタンがクリックされたことを通知
        self.clickedBuy.emit()
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.position_open()
        self.ind_buy.setBuy()

    def on_sell(self):
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 売建ボタンがクリックされたことを通知
        self.clickedSell.emit()
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.position_open()
        self.ind_sell.setSell()

    def on_repay(self):
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 返却ボタンがクリックされたことを通知
        self.clickedRepay.emit()
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.position_close()
        self.ind_buy.setDefault()
        self.ind_sell.setDefault()


class PanelOption(QFrame):
    changedAutoPilotStatus = Signal(bool)
    clickedSave = Signal()
    clickedSetting = Signal()

    def __init__(self, res: AppRes, code: str):
        super().__init__()
        self.res = res
        self.code = code

        self.setFrameStyle(
            QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken
        )
        self.setLineWidth(1)
        layout = HBoxLayout()
        self.setLayout(layout)

        # オートパイロット（自動売買）
        self.autopilot = autopilot = ToggleButtonAutoPilot(res)
        autopilot.setChecked(True)  # デフォルトで ON
        autopilot.toggled.connect(self.toggledAutoPilot)
        layout.addWidget(autopilot)

        pad = PadH()
        layout.addWidget(pad)

        # 設定
        but_setting = ButtonSetting(res)
        but_setting.clicked.connect(self.clickedSetting.emit)
        layout.addWidget(but_setting)

        # チャートの保存
        but_save = ButtonSave(res)
        but_save.clicked.connect(self.clickedSave.emit)
        layout.addWidget(but_save)

    def isAutoPilotEnabled(self) -> bool:
        return self.autopilot.isChecked()

    def setAutoPilotEnabled(self, state: bool = True):
        self.autopilot.setChecked(state)

    def toggledAutoPilot(self, state: bool):
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 AutoPilot 状態の変更を通知するシグナル
        self.changedAutoPilotStatus.emit(state)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
