import logging

from PySide6.QtCore import Signal

from structs.app_enum import PositionType
from structs.res import AppRes
from widgets.docks import DockWidget
from widgets.labels import LCDIntWithTitle, LCDValueWithTitle


class DockTrader(DockWidget):
    clickedBuy = Signal(str, float, str)
    clickedSell = Signal(str, float, str)
    clickedRepay = Signal(str, float, str)
    notifyNewPSARParams = Signal(str, dict)

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

    def setPrice(self, price: float):
        """
        現在株価を表示
        :param price:
        :return:
        """
        self.price.setValue(price)

