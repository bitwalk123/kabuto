import logging

from structs.res import AppRes
from widgets.containers import PanelTrading
from widgets.docks import DockWidget


class DockMatisse(DockWidget):
    def __init__(self, res: AppRes, ticker: str):
        super().__init__(ticker)
        self.logger = logging.getLogger(__name__)

        # 取引用パネル
        self.trading = trading = PanelTrading()
        trading.clickedBuy.connect(self.on_buy)
        trading.clickedSell.connect(self.on_sell)
        trading.clickedRepay.connect(self.on_repay)
        self.layout.addWidget(trading)

    def on_buy(self):
        """
        建玉の買建
        :return:
        """
        self.logger.info("「買建」ボタンがクリックされました。")

    def on_repay(self):
        """
        建玉の返済
        :return:
        """
        self.logger.info("「返済」ボタンがクリックされました。")

    def on_sell(self):
        """
        建玉の売建
        :return:
        """
        self.logger.info("「売建」ボタンがクリックされました。")
