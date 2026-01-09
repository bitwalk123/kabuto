import pandas as pd

from modules.agent import CronAgent


class Explorer:
    def __init__(self, code: str, dict_ts: dict):
        # cron 用エージェントのインスタンス生成
        self.agent = CronAgent(code, dict_ts)

    def run(self, dict_setting: dict, df: pd.DataFrame):
        self.agent.run(dict_setting, df)

    def getObservations(self) -> pd.DataFrame:
        return self.agent.getObservations()

    def getTechnicals(self) -> pd.DataFrame:
        df = self.agent.getTechnicals()
        return df

    def getTransaction(self) -> tuple[pd.DataFrame, int, float]:
        df = self.agent.getTransaction()
        n_trade = len(df)
        total = df["損益"].sum()
        return df, n_trade, total
