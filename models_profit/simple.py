from models_profit.abstract import ProfitSimulatorABS


class ProfitSimulator(ProfitSimulatorABS):
    NAME = "simple"

    def run(self) -> dict:
        print(f"モデル名 : {self.NAME}")

        for r in range(len(self.df)):
            row = self.df.iloc[r]
            ts = row["ts"]
            price = row["price"]
            ma_gc = row["ma_gc"]
            ma_dc = row["ma_dc"]

            if ma_gc > 0:
                note = "ゴールデン・クロス"
                print(ts, price, note)
            if ma_dc > 0:
                note = "デッド・クロス"
                print(ts, price, note)

        return dict()
