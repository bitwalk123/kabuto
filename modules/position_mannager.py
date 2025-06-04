import logging

from PySide6.QtCore import QObject, Signal

from structs.posman import PositionType


class PositionManager(QObject):
    notifyProfit = Signal(str, float)
    threadFinished = Signal(bool)

    def __init__(self, list_ticker: list):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.order = 0  # 注文番号
        self.unit = 100  # 売買単位

        self.dict_price = dict()
        self.dict_total = dict()
        self.dict_position = dict()
        for ticker in list_ticker:
            self.dict_price[ticker] = 0.  # 建玉取得時の株価
            self.dict_total[ticker] = 0.  # 銘柄毎の収益
            self.dict_position[ticker] = PositionType.NONE

    def openPosition(self, ticker: str, price: float, position: PositionType):
        self.dict_price[ticker] = price
        self.dict_position[ticker] = position

    def closePosition(self, ticker: str, price: float):
        if self.dict_position[ticker] == PositionType.BUY:
            self.dict_total[ticker] += price - self.dict_price[ticker]
        elif self.dict_position[ticker] == PositionType.SELL:
            self.dict_total[ticker] += self.dict_price[ticker] - price
        self.dict_price[ticker] = 0
        self.dict_position[ticker] = PositionType.NONE

    def getProfit(self, ticker: str, price: float):
        if self.dict_position[ticker] == PositionType.BUY:
            profit = price - self.dict_price[ticker]
        elif self.dict_position[ticker] == PositionType.SELL:
            profit = self.dict_price[ticker] - price
        else:
            profit = 0
        self.notifyProfit.emit(ticker, profit)
