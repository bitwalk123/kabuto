from models_profit.abstract import ProfitSimulatorABS
from structs.app_enum import ActionType


class ProfitSimulator(ProfitSimulatorABS):
    NAME = "simple"
    DESC = "クロス・シグナル間での単純売買"

    def run(self, progress_callback) -> dict:
        dict_result = dict()
        print(f"モデル名 : {self.NAME}")

        ts = 0
        price = 0
        profit_max = 0

        c_ts = self.df.columns.get_loc("ts")
        c_price = self.df.columns.get_loc("price")
        c_ma_gc = self.df.columns.get_loc("ma_gc")
        c_ma_dc = self.df.columns.get_loc("ma_dc")
        c_profit = self.df.columns.get_loc("profit")
        c_profit_max = self.df.columns.get_loc("profit_max")

        n = len(self.df)
        for r in range(n):
            ts = self.df.iat[r, c_ts]
            price = self.df.iat[r, c_price]
            ma_gc = self.df.iat[r, c_ma_gc]
            ma_dc = self.df.iat[r, c_ma_dc]

            if 0 < ma_gc:
                note = "ゴールデン・クロス"
                if self.posman.hasPosition(self.code):
                    # 返済
                    self.posman.closePosition(self.code, ts, price, note)
                    profit_max = 0
                else:
                    # 買建
                    self.posman.openPosition(self.code, ts, price, ActionType.BUY, note)
            if 0 < ma_dc:
                note = "デッド・クロス"
                if self.posman.hasPosition(self.code):
                    # 返済
                    self.posman.closePosition(self.code, ts, price, note)
                    profit_max = 0
                else:
                    # 売建
                    self.posman.openPosition(self.code, ts, price, ActionType.SELL, note)

            # 含み損益
            profit = self.posman.getProfit(self.code, price)
            if profit_max < profit:
                profit_max = profit
            self.df.iat[r, c_profit] = profit
            self.df.iat[r, c_profit_max] = profit_max

            progress = int((r + 1) / n * 100)
            progress_callback(progress)

        # 含み損益などのティックデータ
        dict_result["tick"] = self.df

        # 取引結果
        if self.posman.hasPosition(self.code):
            self.posman.closePosition(self.code, ts, price, "強制返済")
        dict_result["transaction"] = self.posman.getTransactionResult()

        return dict_result
