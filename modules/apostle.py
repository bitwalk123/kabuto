import os

import pandas as pd

from funcs.ios import load_setting, get_excel_sheet
from modules.agent import CronAgent
from structs.res import AppRes


class Apostle:
    """
    cron で DOE を実行
    """

    def __init__(self):
        self.res = AppRes()
        self.dict_doe = None
        self.name_doe = "doe-8"
        self.code = code = "7011"
        self.agent = CronAgent(code)
        self.factor_doe = ["PERIOD_MA_1", "PERIOD_MA_2"]
        self.df_matrix = pd.DataFrame({
            "PERIOD_MA_1": [
                30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190,
                30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190,
                30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190,
                30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190,
                30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190,
                30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190,
                30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190,
                30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190,
                30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190,
                30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190,
                30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190,
                30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190,
                30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190,
                30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190,
                30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190,
                30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190,
                30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190,
            ],
            "PERIOD_MA_2": [
                300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300,
                350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350, 350,
                400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400,
                450, 450, 450, 450, 450, 450, 450, 450, 450, 450, 450, 450, 450, 450, 450, 450, 450,
                500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
                550, 550, 550, 550, 550, 550, 550, 550, 550, 550, 550, 550, 550, 550, 550, 550, 550,
                600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600,
                650, 650, 650, 650, 650, 650, 650, 650, 650, 650, 650, 650, 650, 650, 650, 650, 650,
                700, 700, 700, 700, 700, 700, 700, 700, 700, 700, 700, 700, 700, 700, 700, 700, 700,
                750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750, 750,
                800, 800, 800, 800, 800, 800, 800, 800, 800, 800, 800, 800, 800, 800, 800, 800, 800,
                850, 850, 850, 850, 850, 850, 850, 850, 850, 850, 850, 850, 850, 850, 850, 850, 850,
                900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900,
                950, 950, 950, 950, 950, 950, 950, 950, 950, 950, 950, 950, 950, 950, 950, 950, 950,
                1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
                1050, 1050, 1050, 1050, 1050, 1050, 1050, 1050, 1050, 1050, 1050, 1050, 1050, 1050, 1050, 1050, 1050,
                1100, 1100, 1100, 1100, 1100, 1100, 1100, 1100, 1100, 1100, 1100, 1100, 1100, 1100, 1100, 1100, 1100,
            ],
        })

    def add_condition_result(self, row_condition: int, path_excel: str, n_trade: int, total: float):
        # file
        key = "file"
        value = os.path.basename(path_excel)
        self.dict_doe.setdefault(key, []).append(value)
        # code
        key = "code"
        value = self.code
        self.dict_doe.setdefault(key, []).append(value)
        # trade
        key = "trade"
        value = n_trade
        self.dict_doe.setdefault(key, []).append(value)
        # total
        key = "total"
        value = total
        self.dict_doe.setdefault(key, []).append(value)
        # Experiment Factors
        for key in self.factor_doe:
            value = self.df_matrix.at[row_condition, key]
            self.dict_doe.setdefault(key, []).append(value)

    def get_file_output(self, file_excel) -> str:
        file_body_without_ext = os.path.splitext(os.path.basename(file_excel))[0]
        path_result = os.path.join(
            self.res.dir_output,
            self.name_doe,
            self.code,
            f"{file_body_without_ext}.csv"
        )
        return path_result

    def run(self):
        files = sorted(os.listdir(self.res.dir_collection))
        dict_setting = load_setting(self.res, self.code)

        for path_excel in files:
            name_dir = os.path.join(self.res.dir_collection, path_excel)
            df = get_excel_sheet(name_dir, self.code)

            self.dict_doe = dict()
            for row_condition in range(len(self.df_matrix)):
                for key in self.factor_doe:
                    dict_setting[key] = int(self.df_matrix.at[row_condition, key])
                n_trade, total = self.agent.run(dict_setting, df)

                self.add_condition_result(row_condition, path_excel, n_trade, total)

            df_doe = pd.DataFrame(self.dict_doe)
            print()
            print(df_doe)
            path_result = self.get_file_output(path_excel)
            # 　ディレクトリが存在していなかったら作成
            path_dir = os.path.dirname(path_result)
            if not os.path.isdir(path_dir):
                os.makedirs(path_dir)
            df_doe.to_csv(path_result, index=False)
            print(f"結果を {path_result} へ保存しました。")
