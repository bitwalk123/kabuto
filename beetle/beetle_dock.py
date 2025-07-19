from PySide6.QtCore import Signal

from structs.res import AppRes
from widgets.containers import PanelTrading, PanelOption
from widgets.docks import DockWidget
from widgets.labels import LCDValueWithTitle, LCDIntWithTitle


class DockBeetleTrader(DockWidget):
    clickedBuy = Signal(str, float, str)
    clickedSell = Signal(str, float, str)
    clickedRepay = Signal(str, float, str)

    def __init__(self, res: AppRes, title: str):
        super().__init__(title)
        self.trend = 0

        # 現在株価
        self.price = price = LCDValueWithTitle("現在株価")
        self.layout.addWidget(price)
        # 含み損益
        self.profit = profit = LCDValueWithTitle("含み損益")
        self.layout.addWidget(profit)
        # 合計収益
        self.total = total = LCDValueWithTitle("合計収益")
        self.layout.addWidget(total)
        # EP 更新回数
        self.epupd = epupd = LCDIntWithTitle("EP 更新回数")
        self.layout.addWidget(epupd)

        # 取引用パネル
        self.trading = trading = PanelTrading()
        self.layout.addWidget(trading)

        # オプションパネル
        self.option = option = PanelOption(res)
        self.layout.addWidget(option)

    def doBuy(self) -> bool:
        """
        「買建」ボタンをクリックして建玉を売る。
        :return:
        """
        if self.trading.buy.isEnabled():
            self.trading.buy.animateClick()
            return True
        else:
            return False

    def doSell(self) -> bool:
        """
        「売建」ボタンをクリックして建玉を売る。
        :return:
        """
        if self.trading.sell.isEnabled():
            self.trading.sell.animateClick()
            return True
        else:
            return False

    def doRepay(self) -> bool:
        """
        「返済」ボタンをクリックして建玉を売る。
        :return:
        """
        if self.trading.repay.isEnabled():
            self.trading.repay.animateClick()
            return True
        else:
            return False

    def finishAutoTrade(self):
        pass

    def setEPUpd(self, epupd: int):
        self.epupd.setValue(epupd)

    def setPrice(self, price: float):
        self.price.setValue(price)

    def setProfit(self, profit: float):
        self.profit.setValue(profit)

    def setTotal(self, total: float):
        self.total.setValue(total)

    def setTrend(self, trend: int, epupd: int):
        if self.trend != trend:
            pass
        self.trend = trend
        self.setEPUpd(epupd)
