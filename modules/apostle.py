import glob
import os

import pandas as pd

from structs.res import AppRes


class Apostle:
    def __init__(self):
        self.res = AppRes()

    def get_sources(self) -> list:
        dir_path = self.res.dir_collection
        list_excel = glob.glob(os.path.join(dir_path, "ticks_*.xlsx"))
        return list_excel

    def run(self):
        # ファイル一覧の取得
        list_excel = self.get_sources()
        for file_excel in list_excel[:1]:
            # Excel ブックの読み込み
            wb = pd.ExcelFile(file_excel)
            # Excel ワークシート名の一覧
            list_sheet = wb.sheet_names
            for sheet in list_sheet[:1]:
                print(sheet)
