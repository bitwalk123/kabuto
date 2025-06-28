import datetime
import glob
import os

import pandas as pd

from funcs.common import get_date_str_from_collections
from funcs.plot import plot_mpl_chart
from funcs.tse import get_ticker_name_list
from modules.simulator import TradeSimulator
from structs.res import AppRes


class Apostle:
    def __init__(self):
        self.res = AppRes()

    def get_sources(self) -> list:
        """
        集計対象のファイルリストを返す
        :return:
        """
        dir_path = self.res.dir_collection
        list_excel = glob.glob(os.path.join(dir_path, "ticks_*.xlsx"))
        return list_excel

    def run(self):
        # ファイル一覧の取得
        list_excel = self.get_sources()
        for file_excel in sorted(list_excel[:1]):
            # 出力先のディレクトリ
            date_str = get_date_str_from_collections(file_excel)
            dir_report = os.path.join(self.res.dir_report, date_str)
            if not os.path.exists(dir_report):
                os.mkdir(dir_report)

            # Excel ブックの読み込み
            wb = pd.ExcelFile(file_excel)

            # Excel ワークシート名の一覧
            list_sheet = wb.sheet_names
            dict_ticker = get_ticker_name_list(list_sheet)
            for sheet in sorted(list_sheet):
                df = pd.read_excel(file_excel, sheet_name=sheet)
                df.index = pd.to_datetime(
                    [datetime.datetime.fromtimestamp(t) for t in df["Time"]]
                )
                # シミュレーション
                dict_conf = {
                    "moving median": 3,
                    "AF": 0.00004,
                    "rolling N": 30,
                }
                simulator = TradeSimulator(sheet, df, dict_conf)
                profit = simulator.run()
                print(sheet, profit)

                # チャート
                title = f"{dict_ticker[sheet]} ({sheet}) on {date_str}"
                condition = (
                    f"moving median = {dict_conf["moving median"]}, "
                    f"AF = {dict_conf["AF"]:.5f}, "
                    f"rolling N = {dict_conf["rolling N"]}, "
                )
                file_img = (
                    f'{sheet}_'
                    f'{dict_conf["moving median"]:02}_'
                    f'{dict_conf["AF"]:.5f}_'
                    f'{dict_conf["rolling N"]:03}'
                    '.png'
                )
                name_img = os.path.join(dir_report, file_img)
                plot_mpl_chart(df, title, condition, name_img)
