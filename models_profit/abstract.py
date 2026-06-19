from abc import ABC, abstractmethod

import pandas as pd


class ProfitSimulatorABS(ABC):
    NAME = "template"

    def __init__(self, df: pd.DataFrame):
        self.df = df

    @abstractmethod
    def run(self) -> dict:
        pass
