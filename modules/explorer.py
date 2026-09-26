import pandas as pd


class Explorer:
    def __init__(self):
        csvname = "doe/doe-001.csv"
        self.df = df = pd.read_csv(csvname)
        self.row_max = len(df)
        self.row_current = 0

    def __iter__(self):
        self.row_current = 0  # ループ開始時にリセット
        return self

    def __next__(self) -> dict:
        if self.row_current < self.row_max:
            dict_condition = self.df.loc[self.row_current].to_dict()
            self.row_current += 1
            return dict_condition
        else:
            raise StopIteration
