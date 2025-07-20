import logging

from PySide6.QtCore import Signal

from beetle.beetle_pacman import PacMan
from structs.posman import PositionType
from structs.res import AppRes
from widgets.containers import PanelTrading, PanelOption
from widgets.docks import DockWidget
from widgets.labels import LCDValueWithTitle, LCDIntWithTitle


class DockBeetleTrader(DockWidget):
    clickedBuy = Signal(str, float, str)
    clickedSell = Signal(str, float, str)
    clickedRepay = Signal(str, float, str)

    def __init__(self, res: AppRes, ticker: str):
        super().__init__(ticker)
        self.logger = logging.getLogger(__name__)
        self.ticker = ticker
        self.pacman = PacMan()

        # 現在株価
        self.price = price = LCDValueWithTitle("現在株価")
        self.layout.addWidget(price)
        # 含み損益
        self.profit = profit = LCDValueWithTitle("含み損益")
        self.layout.addWidget(profit)
        # 合計収益
        self.total = total = LCDValueWithTitle("合計収益")
        self.layout.addWidget(total)

        # 取引用パネル
        self.trading = trading = PanelTrading()
        trading.clickedBuy.connect(self.on_buy)
        trading.clickedSell.connect(self.on_sell)
        trading.clickedRepay.connect(self.on_repay)
        self.layout.addWidget(trading)

        # EP 更新回数
        self.epupd = epupd = LCDIntWithTitle("EP 更新回数")
        self.layout.addWidget(epupd)

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

    def forceStopAutoPilot(self):
        if self.doRepay():
            self.logger.info(f"{__name__}: '{self.ticker}'の強制返済をしました。")
        if self.option.isAutoPilotEnabled():
            self.option.setAutoPilotEnabled(False)
            self.logger.info(f"{__name__}: '{self.ticker}'の Autopilot をオフにしました。")

    def on_buy(self):
        note = ""
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 買建ボタンがクリックされたことを通知
        self.clickedBuy.emit(
            self.ticker, self.price.getValue(), note
        )
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def on_repay(self):
        note = ""
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 返済ボタンがクリックされたことを通知
        self.clickedRepay.emit(
            self.ticker, self.price.getValue(), note
        )
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def on_sell(self):
        note = ""
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 売建ボタンがクリックされたことを通知
        self.clickedSell.emit(
            self.ticker, self.price.getValue(), note
        )
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def setEPUpd(self, epupd: int):
        self.epupd.setValue(epupd)

    def setPrice(self, price: float):
        self.price.setValue(price)

    def setProfit(self, profit: float):
        self.profit.setValue(profit)

    def setTotal(self, total: float):
        self.total.setValue(total)

    def setTrend(self, trend: int, epupd: int):
        self.setEPUpd(epupd)
        if self.option.isAutoPilotEnabled():
            ptype: PositionType = self.pacman.setTrend(trend, epupd)
            if ptype == PositionType.BUY:
                self.doBuy()
            elif ptype == PositionType.SELL:
                self.doSell()
            elif ptype == PositionType.REPAY:
                self.doRepay()
            else:
                pass
