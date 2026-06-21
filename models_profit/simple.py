import pandas as pd

from models_profit.abstract import ProfitSimulatorABS


class ProfitSimulator(ProfitSimulatorABS):
    NAME = "simple"

    def __init__(self, df: pd.DataFrame):
        super().__init__(df)

    def run(self) -> dict:
        print(self.NAME)
        return dict()
