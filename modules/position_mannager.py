import logging

import pandas as pd

from funcs.tide import conv_datetime_from_timestamp
from structs.posman import PositionType


class PositionManager:
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

    def initPosition(self, list_code: list):
        for code in list_code:
            self.dict_price[code] = 0.  # 建玉取得時の株価
            self.dict_total[code] = 0.  # 銘柄毎の収益
            self.dict_position[code] = PositionType.NONE

    def openPosition(self, code: str, ts: float, price: float, position: PositionType, note: str = ""):
        """
        ポジションをオープン（建玉取得）
        :param code:
        :param ts:
        :param price:
        :param position:
        :param note:
        :return:
        """
        self.dict_price[code] = price
        self.dict_position[code] = position

        # 取引履歴
        self.order += 1
        r = len(self.df_order)
        self.df_order.at[r, "注文番号"] = self.order
        self.df_order.at[r, "注文日時"] = conv_datetime_from_timestamp(ts)
        self.df_order.at[r, "銘柄コード"] = code
        if position == PositionType.BUY:
            self.df_order.at[r, "売買"] = "買建"
        elif position == PositionType.SELL:
            self.df_order.at[r, "売買"] = "売建"
        self.df_order.at[r, "約定単価"] = price
        self.df_order.at[r, "約定数量"] = self.unit
        self.df_order.at[r, "備考"] = note

    def closePosition(self, code: str, ts: float, price: float, note: str = ""):
        """
        ポジションをクローズ（建玉返済）
        :param code:
        :param ts:
        :param price:
        :param note:
        :return:
        """
        position = self.dict_position[code]
        profit = 0
        if position == PositionType.BUY:
            profit = (price - self.dict_price[code]) * self.unit
            self.dict_total[code] += profit
        elif position == PositionType.SELL:
            profit = (self.dict_price[code] - price) * self.unit
            self.dict_total[code] += profit
        else:
            return

        # 取引履歴
        self.order += 1
        r = len(self.df_order)
        self.df_order.at[r, "注文番号"] = self.order
        self.df_order.at[r, "注文日時"] = conv_datetime_from_timestamp(ts)
        self.df_order.at[r, "銘柄コード"] = code
        if position == PositionType.BUY:
            self.df_order.at[r, "売買"] = "売埋"
        elif position == PositionType.SELL:
            self.df_order.at[r, "売買"] = "買埋"
        self.df_order.at[r, "約定単価"] = price
        self.df_order.at[r, "約定数量"] = self.unit
        self.df_order.at[r, "損益"] = profit
        self.df_order.at[r, "備考"] = note

        # 売買状態のリセット
        self.dict_price[code] = 0
        self.dict_position[code] = PositionType.NONE

    def getProfit(self, code: str, price: float) -> float:
        if price == 0:
            return 0.
        if self.dict_position[code] == PositionType.BUY:
            return (price - self.dict_price[code]) * self.unit
        elif self.dict_position[code] == PositionType.SELL:
            return (self.dict_price[code] - price) * self.unit
        else:
            return 0.

    def getTotal(self, code: str) -> float:
        return self.dict_total[code]

    def getTransactionResult(self) -> pd.DataFrame:
        return self.df_order
