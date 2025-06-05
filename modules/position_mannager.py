import logging

from PySide6.QtCore import QObject, Signal

from structs.posman import PositionType


class PositionManager(QObject):
    threadFinished = Signal(bool)

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.order = 0  # 注文番号
        self.unit = 100  # 売買単位

    def initPosition(self, list_ticker: list):
        self.dict_price = dict()
        self.dict_total = dict()
        self.dict_position = dict()
        for ticker in list_ticker:
            self.dict_price[ticker] = 0.  # 建玉取得時の株価
            self.dict_total[ticker] = 0.  # 銘柄毎の収益
            self.dict_position[ticker] = PositionType.NONE

    def openPosition(self, ticker: str, ts: float, price: float, position: PositionType):
        """
        ポジションをオープン（建玉取得）
        :param ticker:
        :param ts:
        :param price:
        :param position:
        :return:
        """
        self.dict_price[ticker] = price
        self.dict_position[ticker] = position

    def closePosition(self, ticker: str, ts: float, price: float):
        """
        ポジションをクローズ（建玉返済）
        :param ticker:
        :param ts:
        :param price:
        :return:
        """
        if self.dict_position[ticker] == PositionType.BUY:
            self.dict_total[ticker] += (price - self.dict_price[ticker]) * self.unit
        elif self.dict_position[ticker] == PositionType.SELL:
            self.dict_total[ticker] += (self.dict_price[ticker] - price) * self.unit

        self.dict_price[ticker] = 0
        self.dict_position[ticker] = PositionType.NONE

    def getProfit(self, ticker: str, price: float) -> float:
        if price == 0:
            return 0.
        if self.dict_position[ticker] == PositionType.BUY:
            return (price - self.dict_price[ticker]) * self.unit
        elif self.dict_position[ticker] == PositionType.SELL:
            return (self.dict_price[ticker] - price) * self.unit
        else:
            return 0.

    def getTotal(self, ticker: str) -> float:
        return self.dict_total[ticker]
