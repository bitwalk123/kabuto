import logging

from PySide6.QtCore import Signal, QMargins

from modules.panel import PanelOption, PanelTrading
from structs.res import AppRes
from widgets.dialogs import DlgRepair
from widgets.docks import DockWidget
from widgets.labels import LCDValueWithTitle


class DockTrader(DockWidget):
    clickedBuy = Signal(str, float, str, bool)
    clickedSell = Signal(str, float, str, bool)
    clickedRepay = Signal(str, float, str, bool)
    clickedSave = Signal()
    clickedSetting = Signal()

    def __init__(self, res: AppRes, code: str) -> None:
        super().__init__(code)
        self.logger = logging.getLogger(__name__)
        self.res = res
        self.code = code

        """
        自動オペレーション用フラグ
        マウスで売買ボタンをクリックしたか、
        エージェントが売買シグナルを出したのかを
        区別するためのフラグ
        """
        self.auto = False

        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        #  UI
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        self.setContentsMargins(QMargins(0, 0, 0, 0))

        # 現在株価（表示）
        self.price = price = LCDValueWithTitle("現在株価")
        self.layout.addWidget(price)
        # 含み損益（表示）
        self.profit = profit = LCDValueWithTitle("含み損益")
        self.layout.addWidget(profit)
        # 合計収益（表示）
        self.total = total = LCDValueWithTitle("合計収益")
        self.layout.addWidget(total)

        # ---------------------------------------------------------------------
        # 取引用パネル
        # ---------------------------------------------------------------------
        self.panel_trading = panel_trading = PanelTrading()
        panel_trading.clickedBuy.connect(self.on_buy)
        panel_trading.clickedSell.connect(self.on_sell)
        panel_trading.clickedRepay.connect(self.on_repay)
        self.layout.addWidget(panel_trading)

        # ---------------------------------------------------------------------
        # オプションパネル
        # ---------------------------------------------------------------------
        self.panel_option = panel_option = PanelOption(res, code)
        panel_option.clickedSave.connect(self.on_save)
        panel_option.clickedRepair.connect(self.on_repair)
        panel_option.clickedSetting.connect(self.on_setting)
        self.layout.addWidget(panel_option)

    def force_repay(self) -> None:
        """
        強制返済（取引終了時）
        :return:
        """
        if self.doRepay():
            self.logger.info(f"'{self.code}'の強制返済をしました。")

    def on_buy(self) -> None:
        """
        買建ボタンがクリックされた時の処理
        :return:
        """
        note: str = ""
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 買建ボタンがクリックされたことを通知
        self.clickedBuy.emit(
            self.code, self.price.getValue(), note, self.auto
        )
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.auto = False

    def on_sell(self) -> None:
        """
        売建ボタンがクリックされた時の処理
        :return:
        """
        note: str = ""
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 売建ボタンがクリックされたことを通知
        self.clickedSell.emit(
            self.code, self.price.getValue(), note, self.auto
        )
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.auto = False

    def on_repay(self) -> None:
        """
        返済ボタンがクリックされた時の処理
        :return:
        """
        note: str = ""
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 返済ボタンがクリックされたことを通知
        self.clickedRepay.emit(
            self.code, self.price.getValue(), note, self.auto
        )
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.auto = False

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

    """
    def isDisparityChecked(self) -> bool:
        return self.panel_option.disparity.isEnabled()
    """

    def setPrice(self, price: float) -> None:
        """
        現在株価を表示
        :param price:
        :return:
        """
        self.price.setValue(price)

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
    def doBuy(self) -> bool:
        """
        「買建」ボタンをクリックして建玉を売る。
        :return:
        """
        if self.panel_trading.buy.isEnabled():
            self.auto = True
            self.panel_trading.buy.animateClick()
            return True
        else:
            self.auto = False
            return False

    def doSell(self) -> bool:
        """
        「売建」ボタンをクリックして建玉を売る。
        :return:
        """
        if self.panel_trading.sell.isEnabled():
            self.auto = True
            self.panel_trading.sell.animateClick()
            return True
        else:
            self.auto = False
            return False

    def doRepay(self) -> bool:
        """
        「返済」ボタンをクリックして建玉を売る。
        :return:
        """
        if self.panel_trading.repay.isEnabled():
            self.auto = True
            self.panel_trading.repay.animateClick()
            return True
        else:
            self.auto = False
            return False

    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
    # 実売買移行用
    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
    def next_trading_buttons_status(self, price: float) -> bool:
        return self.panel_trading.set_next_status(price)
