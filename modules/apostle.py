import datetime
import os

import pandas as pd

from funcs.common import get_date_str_from_collections, get_sources_for_collection
from funcs.plot import plot_mpl_chart
from funcs.tse import get_ticker_name_list
from modules.simulator import TradeSimulator
from structs.res import AppRes


class Apostle:
    def __init__(self):
        self.res = AppRes()

    def run(self):
        # ファイル一覧の取得
        list_excel = get_sources_for_collection(self.res.dir_collection)
        # 日付（Excel ファイル）毎ループ
        file_excel = list_excel[-1]
        print(file_excel)
        # 出力先のディレクトリ
        date_str = get_date_str_from_collections(file_excel)
        dir_report = os.path.join(self.res.dir_report, date_str)
        if not os.path.exists(dir_report):
            os.mkdir(dir_report)

        df_result = pd.DataFrame({
            "日付": [],
            "銘柄コード": [],
            "移動メディアン数": [],
            "加速因数": [],
            "多数決数": [],
            "損益": [],
        })
        df_result = df_result.astype(object)

        # Excel ブックの読み込み
        wb = pd.ExcelFile(file_excel)

        # Excel ワークシート名の一覧
        list_sheet = wb.sheet_names
        dict_ticker = get_ticker_name_list(list_sheet)

        # 銘柄（sheet名）毎ループ
        for sheet in sorted(list_sheet):
            df = pd.read_excel(file_excel, sheet_name=sheet)
            df.index = pd.to_datetime(
                [datetime.datetime.fromtimestamp(t) for t in df["Time"]]
            )
            # シミュレーション
            for mm in [3, 6]:
                for af in [0.000025, 0.00005, 0.000075, 0.0001]:
                    for rn in [30, 60]:
                        dict_conf = {
                            "moving median": mm,
                            "AF": af,
                            "rolling N": rn,
                        }
                        self.simulator = TradeSimulator(sheet, df, dict_conf)
                        profit = self.simulator.run()
                        row = len(df_result)
                        df_result.at[row, "日付"] = date_str
                        df_result.at[row, "銘柄コード"] = sheet
                        df_result.at[row, "移動メディアン数"] = mm
                        df_result.at[row, "加速因数"] = af
                        df_result.at[row, "多数決数"] = rn
                        df_result.at[row, "損益"] = profit

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

        # 結果の出力
        print(df_result)
        name_report = os.path.join(dir_report, f"report_{date_str}.csv")
        df_result.to_csv(name_report, index=False)
