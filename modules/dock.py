import logging

from PySide6.QtCore import Signal, QMargins

from modules.panel import (
    PanelControl,
    PanelOption,
    PanelTrading,
)
from structs.res import AppRes
from widgets.dialogs import DlgRepair
from widgets.docks import DockWidget
from widgets.labels import LCDValueWithTitle


class DockTrader(DockWidget):
    clickedBuy = Signal(str, float, str)
    clickedSell = Signal(str, float, str)
    clickedRepay = Signal(str, float, str)
    clickedSave = Signal()
    clickedSetting = Signal()
    changedAutoPilot = Signal(bool)
    notifyStatusCross = Signal(bool)

    def __init__(self, res: AppRes, code: str) -> None:
        super().__init__(code)
        self.logger = logging.getLogger(__name__)
        self.res = res
        self.code = code
        self.note = ""

        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        #  UI
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        self.setContentsMargins(QMargins(0, 0, 0, 0))

        # 現在株価（表示）
        self.price = price = LCDValueWithTitle("現在株価")
        self.layout.addWidget(price)
        # VWAP（表示）
        self.vwap = vwap = LCDValueWithTitle("VWAP")
        self.layout.addWidget(vwap)
        # 長周期移動平均（表示）
        self.ma2 = ma_2 = LCDValueWithTitle("移動平均")
        self.layout.addWidget(ma_2)
        # 含み損益（表示）
        self.profit = profit = LCDValueWithTitle("含み損益")
        self.layout.addWidget(profit)
        # 合計収益（表示）
        self.total = total = LCDValueWithTitle("合計収益")
        self.layout.addWidget(total)

        # ---------------------------------------------------------------------
        # オプション・パネル
        # ---------------------------------------------------------------------
        self.panel_option = panel_option = PanelOption(res, code)
        panel_option.clickedSave.connect(self.on_save)
        panel_option.clickedRepair.connect(self.on_repair)
        panel_option.clickedSetting.connect(self.on_setting)
        panel_option.toggledAutoPilot.connect(self.on_autopilot)
        self.layout.addWidget(panel_option)

        # ---------------------------------------------------------------------
        # コントロール用パネル
        # ---------------------------------------------------------------------
        self.panel_control = panel_control = PanelControl()
        panel_control.changedStatusCross.connect(self.on_status_cross_changed)
        self.layout.addWidget(panel_control)

        # ---------------------------------------------------------------------
        # トレーディング用パネル
        # ---------------------------------------------------------------------
        self.panel_trading = panel_trading = PanelTrading()
        panel_trading.clickedBuy.connect(self.on_buy)
        panel_trading.clickedSell.connect(self.on_sell)
        panel_trading.clickedRepay.connect(self.on_repay)
        self.layout.addWidget(panel_trading)

    def force_repay(self) -> None:
        """
        強制返済（取引終了時）
        :return:
        """
        if self.doRepay({"reason": "強制返済"}):
            self.logger.info(f"'{self.code}'の強制返済をしました。")

    def on_autopilot(self, flag):
        self.changedAutoPilot.emit(flag)

    def on_buy(self) -> None:
        """
        買建ボタンがクリックされた時の処理
        :return:
        """
        if self.note == "":
            self.note = "手動で買建"
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 買建ボタンがクリックされたことを通知
        self.clickedBuy.emit(self.code, self.price.getValue(), self.note)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.note = ""

    def on_sell(self) -> None:
        """
        売建ボタンがクリックされた時の処理
        :return:
        """
        if self.note == "":
            self.note = "手動で売建"
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 売建ボタンがクリックされたことを通知
        self.clickedSell.emit(self.code, self.price.getValue(), self.note)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.note = ""

    def on_repay(self) -> None:
        """
        返済ボタンがクリックされた時の処理
        :return:
        """
        if self.note == "":
            self.note = "手動で返済"
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 返済ボタンがクリックされたことを通知
        self.clickedRepay.emit(self.code, self.price.getValue(), self.note)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.note = ""

    def on_repair(self) -> None:
        dlg = DlgRepair(self.res)
        if dlg.exec():
            flag: bool = dlg.getStatus()
            self.panel_trading.switchActivate(flag)
        else:
            return

    def on_save(self) -> None:
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 保存ボタンがクリックされたことを通知
        self.clickedSave.emit()
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def on_setting(self):
        self.clickedSetting.emit()

    def on_status_cross_changed(self, state: bool):
        self.notifyStatusCross.emit(state)

    def setPrice(self, price: float) -> None:
        """
        現在株価を表示
        :param price:
        :return:
        """
        self.price.setValue(price)

    def setMA2(self, ma2: float) -> None:
        """
        VWAPを表示
        :param ma2:
        :return:
        """
        self.ma2.setValue(ma2)

    def setVWAP(self, vwap: float) -> None:
        """
        VWAPを表示
        :param vwap:
        :return:
        """
        self.vwap.setValue(vwap)

    def setProfit(self, profit: float) -> None:
        """
        現在の含み益を表示
        :param profit:
        :return:
        """
        self.profit.setValue(profit)

    def setTotal(self, total: float) -> None:
        """
        現在の損益合計を表示
        :param total:
        :return:
        """
        self.total.setValue(total)

    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
    # Agent からのアクション
    # 手動でボタンをクリックした時と区別できるようにする
    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
    def doBuy(self, states: dict) -> bool:
        """
        「買建」ボタンをクリックして建玉を売る。
        :param states:
        :return:
        """
        if "reason" in states:
            self.note = states["reason"]
        if self.panel_trading.buy.isEnabled():
            self.panel_trading.buy.animateClick()
            return True
        else:
            return False

    def doSell(self, states: dict) -> bool:
        """
        「売建」ボタンをクリックして建玉を売る。
        :param states:
        :return:
        """
        if "reason" in states:
            self.note = states["reason"]
        if self.panel_trading.sell.isEnabled():
            self.panel_trading.sell.animateClick()
            return True
        else:
            return False

    def doRepay(self, states: dict) -> bool:
        """
        「返済」ボタンをクリックして建玉を売る。
        :param states:
        :return:
        """
        if "reason" in states:
            self.note = states["reason"]
        if self.panel_trading.repay.isEnabled():
            self.panel_trading.repay.animateClick()
            return True
        else:
            return False

    def setAutoPilotDisabled(self):
        """
        AutoPilot ボタンがチェックされていたらチェックを外す
        :return:
        """
        if self.panel_option.but_autopilot.isChecked():
            self.panel_option.but_autopilot.animateClick()

    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
    # 実売買移行用
    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
    def next_trading_buttons_status(self, price: float) -> bool:
        return self.panel_trading.set_next_status(price)
