import datetime
import logging

import pandas as pd

from structs.app_enum import ActionType


class PositionManager:
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.order = 0  # 注文番号
        self.unit = 100  # 売買単位

        self.dict_price = dict()
        self.dict_total = dict()
        self.dict_action = dict()

        # 取引履歴用辞書
        self.records = {
            "注文番号": [],
            "注文日時": [],
            "銘柄コード": [],
            "売買": [],
            "約定単価": [],
            "約定数量": [],
            "損益": [],
            "備考": [],
        }

    def initPosition(self, list_code: list):
        for code in list_code:
            self.dict_price[code] = 0.  # 建玉取得時の株価
            self.dict_total[code] = 0.  # 銘柄毎の収益
            self.dict_action[code] = ActionType.HOLD

    def openPosition(self, code: str, ts: float, price: float, action: ActionType, note: str = ""):
        """
        ポジションをオープン（建玉取得）
        :param code:
        :param ts:
        :param price:
        :param action:
        :param note:
        :return:
        """
        self.dict_price[code] = price
        self.dict_action[code] = action

        # 取引履歴
        self.order += 1
        self.records["注文番号"].append(self.order)
        self.records["注文日時"].append(ts)
        self.records["銘柄コード"].append(code)
        if action == ActionType.BUY:
            self.records["売買"].append("買建")
        elif action == ActionType.SELL:
            self.records["売買"].append("売建")
        self.records["約定単価"].append(price)
        self.records["約定数量"].append(self.unit)
        self.records["損益"].append(None)
        self.records["備考"].append(note)

    def closePosition(self, code: str, ts: float, price: float, note: str = ""):
        """
        ポジションをクローズ（建玉返済）
        :param code:
        :param ts:
        :param price:
        :param note:
        :return:
        """
        action = self.dict_action[code]
        profit = 0
        if action == ActionType.BUY:
            profit = (price - self.dict_price[code]) * self.unit
            self.dict_total[code] += profit
        elif action == ActionType.SELL:
            profit = (self.dict_price[code] - price) * self.unit
            self.dict_total[code] += profit
        else:
            return

        # 取引履歴
        self.order += 1
        self.records["注文番号"].append(self.order)
        self.records["注文日時"].append(ts)
        self.records["銘柄コード"].append(code)
        if action == ActionType.BUY:
            self.records["売買"].append("売埋")
        elif action == ActionType.SELL:
            self.records["売買"].append("買埋")
        self.records["約定単価"].append(price)
        self.records["約定数量"].append(self.unit)
        self.records["損益"].append(profit)
        self.records["備考"].append(note)

        # 売買状態のリセット
        self.dict_price[code] = 0
        self.dict_action[code] = ActionType.HOLD

    def getProfit(self, code: str, price: float) -> float:
        if price == 0:
            return 0.
        if self.dict_action[code] == ActionType.BUY:
            return (price - self.dict_price[code]) * self.unit
        elif self.dict_action[code] == ActionType.SELL:
            return (self.dict_price[code] - price) * self.unit
        else:
            return 0.

    def getTotal(self, code: str) -> float:
        return self.dict_total[code]

    def getTransactionResult(self) -> pd.DataFrame:
        #td = datetime.timedelta(hours=9)
        df = pd.DataFrame(self.records)
        #df["注文日時"] = pd.to_datetime(df["注文日時"], unit="s")
        df["注文日時"] = (
            pd.to_datetime(df["注文日時"], unit="s", utc=True)
            .dt.tz_convert("Asia/Tokyo")
        )
        return df
