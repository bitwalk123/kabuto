from pathlib import Path

import pandas as pd

from structs.file_path import FilePath


class Explorer:
    def __init__(self, csvname: str, num_condition: int):
        # 条件表
        self.dir_result = str(Path(csvname).with_suffix(""))
        self.df = pd.read_csv(csvname)
        self.row_max = len(self.df)

        # 条件指定の場合、指定の条件のみ抜き取り
        if 0 <= num_condition < self.row_max:
            self.df = pd.DataFrame(
                self.df.iloc[num_condition]
            ).T
            self.row_max = len(self.df)
            print(self.df)

        # 現在行位置
        self.row_current = 0

        # 結果用
        self.df_summary = self.df.copy()

    def __iter__(self):
        self.row_current = 0  # ループ開始時にリセット
        return self

    def __next__(self) -> dict:
        if self.row_current < self.row_max:
            dict_condition = self.df.iloc[self.row_current].to_dict()
            return dict_condition
        else:
            raise StopIteration

    def append_result(self, dict_result: dict):
        self.df_summary.loc[self.row_current, ["Profit", "Transactions"]] = dict_result
        self.row_current += 1  # ここでカウンタをインクリメント

    def get_summary(self) -> pd.DataFrame:
        return self.df_summary


class Lister:
    def __init__(self, list_file: list[FilePath]):
        # ファイルリスト
        self.list_file = list_file
        self.row_max = len(self.list_file)

        # 現在行位置
        self.row_current = 0

    def __iter__(self):
        self.row_current = 0  # ループ開始時にリセット
        return self

    def __next__(self) -> FilePath:
        if self.row_current < self.row_max:
            path_file = self.list_file[self.row_current]
            self.row_current += 1  # カウンタをインクリメント
            return path_file
        else:
            raise StopIteration
