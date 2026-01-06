import os

import pandas as pd

from funcs.ios import get_excel_sheet
from funcs.setting import load_setting
from funcs.tide import get_intraday_timestamp
from modules.agent import CronAgent
from structs.res import AppRes


class Disciple:
    """
    json をパラメータ設定を読み込み、
    そのパラメータに従って、指定ファイル、指定銘柄の
    取引シミュレーションを実行
    """

    def __init__(self, excel:str, code:str):
        self.res = res = AppRes()
        self.path_excel = os.path.join(res.dir_collection, excel)
        self.code = code

        # ザラ場の開始時間などのタイムスタンプ取得（Excelの日付）
        dict_ts = get_intraday_timestamp(self.path_excel)

        self.agent = CronAgent(code, dict_ts)

    def run(self):
        dict_setting = load_setting(self.res, self.code)
        df = get_excel_sheet(self.path_excel, self.code)
        self.agent.run(dict_setting, df)

    def getTechnicals(self) -> pd.DataFrame:
        return self.agent.getTechnicals()

    def getTransaction(self) -> pd.DataFrame:
        return self.agent.getTransaction()
