import datetime
import glob
import os

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
import pandas as pd

from funcs.common import get_date_str_from_collections
from funcs.tse import get_ticker_name_list
from structs.res import AppRes


def plot_chart(df: pd.DataFrame, title: str, imgname: str):
    FONT_PATH = "fonts/RictyDiminished-Regular.ttf"
    fm.fontManager.addfont(FONT_PATH)

    # FontPropertiesオブジェクト生成（名前の取得のため）
    font_prop = fm.FontProperties(fname=FONT_PATH)
    font_prop.get_name()

    plt.rcParams["font.family"] = font_prop.get_name()
    plt.rcParams["font.size"] = 16
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.grid()

    ax.plot(df["Price"], color="gray", linewidth=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(imgname)
    plt.close()


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
            for sheet in sorted(list_sheet[:1]):
                title = f"{dict_ticker[sheet]} ({sheet}) on {date_str}"
                name_img = os.path.join(dir_report, f"{sheet}.png")
                df = pd.read_excel(file_excel, sheet_name=sheet)
                df.index = pd.to_datetime(
                    [datetime.datetime.fromtimestamp(t) for t in df["Time"]]
                )
                plot_chart(df, title, name_img)
