import pandas as pd


class Explorer:
    def __init__(self):
        # 条件表
        csvname = "doe/doe-001.csv"
        self.df = df = pd.read_csv(csvname)
        self.row_max = len(df)
        self.row_current = 0  # 現在行位置
        # 結果用
        self.df_summary = self.df.copy()

    def __iter__(self):
        self.row_current = 0  # ループ開始時にリセット
        return self

    def __next__(self) -> dict:
        if self.row_current < self.row_max:
            dict_condition = self.df.loc[self.row_current].to_dict()
            return dict_condition
        else:
            raise StopIteration

    def append_result(self, dict_result: dict):
        # self.df_summary.loc[self.row_current, "Profit"] = dict_result["Profit"]
        # self.df_summary.loc[self.row_current, "Transactions"] = dict_result["Transactions"]
        self.df_summary.loc[self.row_current, ["Profit", "Transactions"]] = dict_result
        self.row_current += 1  # ここでカウンタをインクリメント

    def get_summary(self) -> pd.DataFrame:
        return self.df_summary
