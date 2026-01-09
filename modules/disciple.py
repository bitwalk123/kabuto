import os

import pandas as pd

from funcs.ios import get_excel_sheet
from funcs.tide import get_intraday_timestamp
from modules.agent import CronAgent
from structs.res import AppRes


class Disciple:
    """
    json をパラメータ設定を読み込み、
    そのパラメータに従って、指定ファイル、指定銘柄の
    取引シミュレーションを実行
    """

    def __init__(self, excel: str, code: str, dict_setting: dict):
        self.res = res = AppRes()
        self.path_excel = os.path.join(res.dir_collection, excel)
        self.code = code
        self.dict_setting = dict_setting

        # ザラ場の開始時間などのタイムスタンプ取得（Excelの日付）
        dict_ts = get_intraday_timestamp(self.path_excel)
        # cron 用エージェントのインスタンス生成
        self.agent = CronAgent(code, dict_ts)

    def run(self):
        df = get_excel_sheet(self.path_excel, self.code)
        self.agent.run(self.dict_setting, df)

    def getObservations(self) -> pd.DataFrame:
        return self.agent.getObservations()

    def getTechnicals(self) -> pd.DataFrame:
        return self.agent.getTechnicals()

    def getTransaction(self) -> pd.DataFrame:
        return self.agent.getTransaction()
