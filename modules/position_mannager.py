import logging

import pandas as pd
from PySide6.QtCore import QObject

from funcs.tide import conv_datetime_from_timestamp
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
            "注文番号": [],
            "注文日時": [],
            "銘柄コード": [],
            "売買": [],
            "約定単価": [],
            "約定数量": [],
            "損益": [],
            "備考": [],
        }
        df = pd.DataFrame.from_dict(dict_columns)
        self.df_order = df.astype(object)

    def initPosition(self, list_ticker: list):
        for ticker in list_ticker:
            self.dict_price[ticker] = 0.  # 建玉取得時の株価
            self.dict_total[ticker] = 0.  # 銘柄毎の収益
            self.dict_position[ticker] = PositionType.NONE

    def openPosition(self, ticker: str, ts: float, price: float, position: PositionType, note: str):
        """
        ポジションをオープン（建玉取得）
        :param ticker:
        :param ts:
        :param price:
        :param position:
        :param note:
        :return:
        """
        self.dict_price[ticker] = price
        self.dict_position[ticker] = position

        # 取引履歴
        self.order += 1
        r = len(self.df_order)
        self.df_order.at[r, "注文番号"] = self.order
        self.df_order.at[r, "注文日時"] = conv_datetime_from_timestamp(ts)
        self.df_order.at[r, "銘柄コード"] = ticker
        if position == PositionType.BUY:
            self.df_order.at[r, "売買"] = "買建"
        elif position == PositionType.SELL:
            self.df_order.at[r, "売買"] = "売建"
        self.df_order.at[r, "約定単価"] = price
        self.df_order.at[r, "約定数量"] = self.unit
        self.df_order.at[r, "備考"] = note

    def closePosition(self, ticker: str, ts: float, price: float, note: str):
        """
        ポジションをクローズ（建玉返済）
        :param ticker:
        :param ts:
        :param price:
        :param note:
        :return:
        """
        position = self.dict_position[ticker]
        profit = 0
        if position == PositionType.BUY:
            profit = (price - self.dict_price[ticker]) * self.unit
            self.dict_total[ticker] += profit
        elif position == PositionType.SELL:
            profit = (self.dict_price[ticker] - price) * self.unit
            self.dict_total[ticker] += profit

        # 取引履歴
        self.order += 1
        r = len(self.df_order)
        self.df_order.at[r, "注文番号"] = self.order
        self.df_order.at[r, "注文日時"] = conv_datetime_from_timestamp(ts)
        self.df_order.at[r, "銘柄コード"] = ticker
        if position == PositionType.BUY:
            self.df_order.at[r, "売買"] = "売埋"
        elif position == PositionType.SELL:
            self.df_order.at[r, "売買"] = "買埋"
        self.df_order.at[r, "約定単価"] = price
        self.df_order.at[r, "約定数量"] = self.unit
        self.df_order.at[r, "損益"] = profit
        self.df_order.at[r, "備考"] = note

        # 売買状態のリセット
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

    def getTransactionResult(self) -> pd.DataFrame:
        return self.df_order
