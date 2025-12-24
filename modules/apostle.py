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

        # ---------------------------------------------------------------------
        # DOE
        self.name_doe = "doe-8"
        # ---------------------------------------------------------------------
        # DOE 水準表を読み込む
        path_doe = os.path.join(self.res.dir_doe, f"{self.name_doe}.csv")
        self.df_matrix = pd.read_csv(path_doe)
        self.factor_doe = list(self.df_matrix.columns)
        self.dict_doe = None

        # 対象銘柄
        self.code = code = "7011"
        # self.code = code = "8306"

        # GUI 無しエージェントのインスタンス
        self.agent = CronAgent(code)

    def add_condition_result(
            self,
            row_condition: int,
            path_excel: str,
            n_trade: int,
            total: float
    ):
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

    def get_file_output(self, path_excel: str, code: str) -> str:
        file_body_without_ext = os.path.splitext(os.path.basename(path_excel))[0]
        # 出力名はティックデータの Excel ファイルの拡張子を csv へ変えたもの
        path_result = os.path.join(
            self.res.dir_output,
            self.name_doe,
            code,
            f"{file_body_without_ext}.csv"
        )
        return path_result

    def main_loop(self, path_excel: str, code: str):
        # 結果の出力先（予め、結果が存在するかどうかを確認する）
        path_result = self.get_file_output(path_excel, code)

        # 　ディレクトリが存在していなかったら作成
        path_dir = os.path.dirname(path_result)
        if not os.path.isdir(path_dir):
            os.makedirs(path_dir)

        # ファイルが存在していればスキップ
        if os.path.exists(path_result):
            return

        # ベースの実験条件をロード
        dict_setting = load_setting(self.res, code)

        # 指定したシート (= code) を読み込む
        df = get_excel_sheet(path_excel, code)

        # 結果格納用辞書をリセット
        self.dict_doe = dict()

        # DOE 条件ループ
        for row_condition in range(len(self.df_matrix)):
            for key in self.factor_doe:
                dict_setting[key] = int(self.df_matrix.at[row_condition, key])
            n_trade, total = self.agent.run(dict_setting, df)
            self.add_condition_result(row_condition, path_excel, n_trade, total)

        df_doe = pd.DataFrame(self.dict_doe)
        print()
        print(df_doe)

        # 結果を出力
        df_doe.to_csv(path_result, index=False)
        print(f"結果が {path_result} に保存されました。")

    def run(self):
        files = sorted(os.listdir(self.res.dir_collection))

        for excel in files:
            path_excel = os.path.join(self.res.dir_collection, excel)

            for code in [self.code]:  # シート名からループを回す仕様に変更予定
                self.main_loop(path_excel, code)
