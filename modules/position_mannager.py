import logging

import pandas as pd
from PySide6.QtCore import QObject

from structs.posman import PositionType


class PositionManager(QObject):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.order = 0  # 注文番号
        self.unit = 100  # 売買単位

        self.dict_price = dict()
        self.dict_total = dict()
        self.dict_position = dict()

        dict_columns = {
            '注文番号': [],
            '注文日時': [],
            '銘柄コード': [],
            '売買': [],
            '約定単価': [],
            '約定数量': [],
            '損益': [],
            '備考': [],
        }
        df = pd.DataFrame.from_dict(dict_columns)
        self.df_order = df.astype(object)

    def initPosition(self, list_ticker: list):
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
        self.order += 1
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
        self.order += 1
        if self.dict_position[ticker] == PositionType.BUY:
            profit = (price - self.dict_price[ticker]) * self.unit
            self.dict_total[ticker] += profit
        elif self.dict_position[ticker] == PositionType.SELL:
            profit = (self.dict_price[ticker] - price) * self.unit
            self.dict_total[ticker] += profit

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
