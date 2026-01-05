import os

from funcs.ios import get_excel_sheet
from funcs.setting import load_setting
from modules.agent import CronAgent
from structs.res import AppRes


class Disciple:
    """
    json をパラメータ設定を読み込み、
    そのパラメータに従って、指定ファイル、指定銘柄の
    取引シミュレーションを実行
    """

    def __init__(self):
        self.res = res = AppRes()
        self.code = code = "7011"
        excel = "ticks_20260105.xlsx"
        self.path_excel = os.path.join(res.dir_collection, excel)

        self.agent = CronAgent(code)

    def run(self):
        dict_setting = load_setting(self.res, self.code)
        df = get_excel_sheet(self.path_excel, self.code)
        n_trade, total = self.agent.run(dict_setting, df)
