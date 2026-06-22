from models_profit.abstract import ProfitSimulatorABS
from structs.app_enum import ActionType


class ProfitSimulator(ProfitSimulatorABS):
    NAME = "simple"
    DESC = "クロス・シグナル間での単純売買"

    def run(self) -> dict:
        dict_result = dict()
        print(f"モデル名 : {self.NAME}")

        ts = 0
        price = 0
        for r in range(len(self.df)):
            row = self.df.iloc[r]
            ts = row["ts"]
            price = row["price"]
            ma_gc = row["ma_gc"]
            ma_dc = row["ma_dc"]

            if 0 < ma_gc:
                note = "ゴールデン・クロス"
                if self.posman.hasPosition(self.code):
                    self.posman.closePosition(self.code, ts, price, note)
                else:
                    self.posman.openPosition(self.code, ts, price, ActionType.BUY, note)
            if 0 < ma_dc:
                note = "デッド・クロス"
                if self.posman.hasPosition(self.code):
                    self.posman.closePosition(self.code, ts, price, note)
                else:
                    self.posman.openPosition(self.code, ts, price, ActionType.SELL, note)

        # 取引結果
        if self.posman.hasPosition(self.code):
            self.posman.closePosition(self.code, ts, price, "強制返済")
        dict_result["transaction"] = self.posman.getTransactionResult()

        return dict_result
