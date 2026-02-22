from PySide6.QtCore import QMargins, Signal
from PySide6.QtWidgets import QFrame, QButtonGroup

from structs.res import AppRes
from widgets.buttons import (
    ButtonRepair,
    ButtonSave,
    ButtonSetting,
    ToggleButtonSmall,
    TradeButton,
)
from widgets.containers import (
    IndicatorBuySell,
    PadH,
    Widget, NarrowLine,
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

    def __init__(self) -> None:
        super().__init__()
        self.flag_next_status: bool = True
        self.flag_disabled: bool = True  # 全ての売買・返済ボタンを無効状態フラグ
        self.setContentsMargins(QMargins(0, 0, 0, 0))

        layout = GridLayout()
        layout.setSpacing(0)
        self.setLayout(layout)

        row: int = 0
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

        row += 1
        line = NarrowLine()
        layout.addWidget(line, row, 0, 1, 2)

        row += 1
        # 基準線（相対）
        self.rel = but_rel = ToggleButtonSmall("rel")
        layout.addWidget(but_rel, row, 0)

        # 基準線（絶対）
        self.abs = but_abs = ToggleButtonSmall("abs")
        but_abs.setChecked(True)
        layout.addWidget(but_abs, row, 1)

        # ボタングループ
        self.baseline = baseline = QButtonGroup()
        baseline.addButton(but_rel)
        baseline.addButton(but_abs)

        # 初期状態ではポジション無し
        self.switchDeactivateAll()

    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
    # 売買イベント
    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
    def set_next_status(self, status: bool) -> None:
        if status:
            self.switchActivate(self.flag_next_status)
        else:
            self.switchActivate(not self.flag_next_status)

    def request_buy(self) -> None:
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 買建ボタンがクリックされたことを通知
        self.clickedBuy.emit()
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.switchDeactivateAll()
        self.flag_next_status = False
        self.ind_buy.setBuy()

    def request_sell(self) -> None:
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 売建ボタンがクリックされたことを通知
        self.clickedSell.emit()
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.switchDeactivateAll()
        self.flag_next_status = False
        self.ind_sell.setSell()

    def request_repay(self) -> None:
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 返却ボタンがクリックされたことを通知
        self.clickedRepay.emit()
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.switchDeactivateAll()
        self.flag_next_status = True
        self.ind_buy.setDefault()
        self.ind_sell.setDefault()

    def switchDeactivateAll(self) -> None:
        self.buy.setDisabled(True)
        self.sell.setDisabled(True)
        self.repay.setDisabled(True)

    def switchActivate(self, state: bool) -> None:
        self.buy.setEnabled(state)
        self.sell.setEnabled(state)
        self.repay.setDisabled(state)
        if state:
            self.ind_buy.setDefault()
            self.ind_sell.setDefault()

    def lockButtons(self) -> None:
        if not self.flag_disabled:
            self.flag_disabled = True
            self.switchDeactivateAll()

    def unLockButtons(self) -> None:
        if self.flag_disabled:
            self.flag_disabled = False
            self.switchActivate(True)


class PanelOption(QFrame):
    clickedSave = Signal()
    clickedSetting = Signal()
    clickedRepair = Signal()
    changedDisparity = Signal(bool)

    def __init__(self, res: AppRes, code: str) -> None:
        super().__init__()
        self.res = res
        self.code = code

        self.setFrameStyle(
            QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken
        )
        self.setLineWidth(1)
        layout = HBoxLayout()
        self.setLayout(layout)

        # VWAP との乖離を表示するかどうかのスイッチ
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
