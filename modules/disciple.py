import os

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

    def __init__(self):
        self.res = res = AppRes()
        self.code = code = "7011"
        excel = "ticks_20260105.xlsx"
        self.path_excel = os.path.join(res.dir_collection, excel)

        # ザラ場の開始時間などのタイムスタンプ取得（Excelの日付）
        dict_ts = get_intraday_timestamp(self.path_excel)

        self.agent = CronAgent(code, dict_ts)

    def run(self):
        dict_setting = load_setting(self.res, self.code)
        df = get_excel_sheet(self.path_excel, self.code)
        self.agent.run(dict_setting, df)

        df_transaction = self.agent.getTransaction()
        print("\n取引明細")
        print(df_transaction)
        n_trade = len(df_transaction)
        total = df_transaction["損益"].sum()
        print(f"取引回数: {n_trade} 回 / 総収益: {total} 円/株")

        print("\nテクニカル指標")
        df_technical = self.agent.getTechnicals()
        print(df_technical)