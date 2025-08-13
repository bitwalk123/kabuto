import logging

from PySide6.QtCore import Signal

from modules.panel import PanelTrading
from structs.app_enum import PositionType
from structs.res import AppRes
from widgets.docks import DockWidget
from widgets.labels import LCDValueWithTitle


class DockTrader(DockWidget):
    clickedBuy = Signal(str, float, str)
    clickedSell = Signal(str, float, str)
    clickedRepay = Signal(str, float, str)

    def __init__(self, res: AppRes, code: str):
        super().__init__(code)
        self.logger = logging.getLogger(__name__)
        self.code = code

        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        #  UI
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_

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
        self.trading = trading = PanelTrading()
        trading.clickedBuy.connect(self.on_buy)
        trading.clickedSell.connect(self.on_sell)
        trading.clickedRepay.connect(self.on_repay)
        self.layout.addWidget(trading)

    def on_buy(self):
        """
        買建ボタンがクリックされた時の処理
        :return:
        """
        note = ""
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 買建ボタンがクリックされたことを通知
        self.clickedBuy.emit(
            self.code, self.price.getValue(), note
        )
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def on_repay(self):
        """
        返済ボタンがクリックされた時の処理
        :return:
        """
        note = ""
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 返済ボタンがクリックされたことを通知
        self.clickedRepay.emit(
            self.code, self.price.getValue(), note
        )
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def on_sell(self):
        """
        売建ボタンがクリックされた時の処理
        :return:
        """
        note = ""
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 売建ボタンがクリックされたことを通知
        self.clickedSell.emit(
            self.code, self.price.getValue(), note
        )
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def setPrice(self, price: float):
        """
        現在株価を表示
        :param price:
        :return:
        """
        self.price.setValue(price)

    def setProfit(self, profit: float):
        """
        現在の含み益を表示
        :param profit:
        :return:
        """
        self.profit.setValue(profit)

    def setTotal(self, total: float):
        """
        現在の損益合計を表示
        :param total:
        :return:
        """
        self.total.setValue(total)
