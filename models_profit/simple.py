from models_profit.abstract import ProfitSimulatorABS
from structs.app_enum import ActionType


class ProfitSimulator(ProfitSimulatorABS):
    NAME = "simple"

    def run(self) -> dict:
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
        df_transaction = self.posman.getTransactionResult()
        pnl = df_transaction["損益"].sum()

        print(df_transaction)
        print(f"損益 : {pnl} 円/株")

        return dict()
