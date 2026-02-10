from PySide6.QtCore import QMargins, Signal
from PySide6.QtWidgets import QFrame

from structs.res import AppRes
from widgets.buttons import (
    ButtonSave,
    ButtonSetting,
    TradeButton, ButtonRepair,
)
from widgets.containers import (
    IndicatorBuySell,
    Widget, PadH,
)
from widgets.layouts import (
    GridLayout,
    HBoxLayout,
)
from widgets.switches import Switch


class PanelTrading(Widget):
    """
    トレーディング用パネル
    固定株数でナンピンしない取引が前提
    """
    clickedBuy = Signal()
    clickedRepay = Signal()
    clickedSell = Signal()

    def __init__(self):
        super().__init__()
        self.flag_next_status = None
        self.flag_disabled = True  # 全ての売買・返済ボタンを無効状態フラグ
        self.setContentsMargins(QMargins(0, 0, 0, 0))

        layout = GridLayout()
        layout.setSpacing(0)
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
        but_sell.clicked.connect(self.request_sell)
        layout.addWidget(but_sell, row, 0)

        # 建玉の買建
        self.buy = but_buy = TradeButton("buy")
        but_buy.clicked.connect(self.request_buy)
        layout.addWidget(but_buy, row, 1)

        row += 1
        # 建玉の返却
        self.repay = but_repay = TradeButton("repay")
        but_repay.clicked.connect(self.request_repay)
        layout.addWidget(but_repay, row, 0, 1, 2)

        # 初期状態ではポジション無し
        self.switchDeactivateAll()

    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
    # 売買イベント
    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
    def receive_result(self, status: bool):
        if self.flag_next_status is None:
            # 初期状態で誤って呼ばれた場合の保険
            self.switchActivate(True)
            return
        if status:
            self.switchActivate(self.flag_next_status)
        else:
            self.switchActivate(not self.flag_next_status)

    def request_buy(self):
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 買建ボタンがクリックされたことを通知
        self.clickedBuy.emit()
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.switchDeactivateAll()
        self.flag_next_status = False
        self.ind_buy.setBuy()

    def request_sell(self):
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 売建ボタンがクリックされたことを通知
        self.clickedSell.emit()
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.switchDeactivateAll()
        self.flag_next_status = False
        self.ind_sell.setSell()

    def request_repay(self):
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 返却ボタンがクリックされたことを通知
        self.clickedRepay.emit()
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.switchDeactivateAll()
        self.flag_next_status = True
        self.ind_buy.setDefault()
        self.ind_sell.setDefault()

    def switchDeactivateAll(self):
        self.buy.setDisabled(True)
        self.sell.setDisabled(True)
        self.repay.setDisabled(True)

    def switchActivate(self, state: bool):
        self.buy.setEnabled(state)
        self.sell.setEnabled(state)
        self.repay.setDisabled(state)
        if state:
            self.ind_buy.setDefault()
            self.ind_sell.setDefault()

    def lockButtons(self):
        if not self.flag_disabled:
            self.flag_disabled = True
            self.switchDeactivateAll()

    def unLockButtons(self):
        if self.flag_disabled:
            self.flag_disabled = False
            self.switchActivate(True)


class PanelOption(QFrame):
    clickedSave = Signal()
    clickedSetting = Signal()
    clickedRepair = Signal()
    changedDisparity = Signal(bool)

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

        # 乖離を表示するかどうかのスイッチ
        self.disparity = disparity = Switch()
        disparity.set(False)
        disparity.statusChanged.connect(self.changedDisparity.emit)
        layout.addWidget(disparity)

        pad = PadH()
        layout.addWidget(pad)

        # 売買ボタンの状態修正
        but_repair = ButtonRepair(res)
        but_repair.setToolTip("売買ボタンの状態修正")
        but_repair.clicked.connect(self.clickedRepair.emit)
        layout.addWidget(but_repair)

        # 設定
        but_setting = ButtonSetting(res)
        but_setting.setToolTip("設定")
        but_setting.clicked.connect(self.clickedSetting.emit)
        layout.addWidget(but_setting)

        # チャートの保存
        but_save = ButtonSave(res)
        but_save.setToolTip("チャートの保存")
        but_save.clicked.connect(self.clickedSave.emit)
        layout.addWidget(but_save)
