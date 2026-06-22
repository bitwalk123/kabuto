from abc import ABC, abstractmethod

import pandas as pd

from modules.posman import PositionManager


class ProfitSimulatorABS(ABC):
    NAME = "template"

    def __init__(self, code: str, df: pd.DataFrame):
        self.code = code
        self.df = df

        # ポジション・マネージャ
        self.posman = PositionManager()
        self.posman.initPosition([code])

    @abstractmethod
    def run(self) -> dict:
        pass
