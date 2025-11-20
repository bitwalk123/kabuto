import logging
import os
from time import perf_counter

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QMainWindow

from funcs.ios import get_excel_sheet
from modules.agent import WorkerAgent
from modules.toolbar import ToolBarProphet
from structs.res import AppRes


class Prophet(QMainWindow):
    __app_name__ = "Prophet"
    __version__ = "0.0.1"
    __author__ = "Fuhito Suguri"
    __license__ = "MIT"

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)  # モジュール固有のロガーを取得
        self.res = res = AppRes()

        self.setWindowIcon(QIcon(os.path.join(res.dir_image, "inference.png")))
        title_win = f"{self.__app_name__} - {self.__version__}"
        self.setWindowTitle(title_win)

        self.toolbar = toolbar = ToolBarProphet(res)
        toolbar.clickedPlay.connect(self.on_start)
        self.addToolBar(toolbar)

    def on_start(self):
        dict_info = self.toolbar.getInfo()
        path_model = dict_info["path_model"]
        path_excel = dict_info["path_excel"]
        code = dict_info["code"]

        print("下記の条件で推論を実施します。")
        print(f"モデル\t\t: {path_model}")
        print(f"ティックデータ\t: {path_excel}")
        print(f"銘柄コード\t: {code}")

        # Excel ファイルをデータフレームに読み込む
        df = get_excel_sheet(path_excel, code)
        print("\nExcel ファイルをデータフレームに読み込みました。")
        print(df)

        print("\nAgent のインスタンスを生成します。")
        agent = WorkerAgent(path_model, True)

        print("\n推論ループを開始します。")
        t_start = perf_counter()
        row = 0
        done = False # 推論終了フラグ
        while not done:
            ts = df["Time"].iloc[row]
            price = df["Price"].iloc[row]
            volume = df["Volume"].iloc[row]
            done = agent.addData(ts, price, volume)
            row += 1
            if row > len(df):
                done = True
        # ループ終了時刻
        t_end = perf_counter()

        print("\n推論ループを終了しました。")
        t_delta = t_end - t_start
        print(f"計測時間 :\t\t\t{t_delta:,.3f} sec")
        print(f"ティック数 :\t\t\t{row:,d} ticks")
        print(f"処理時間 / ティック :\t{t_delta / row * 1_000:.3f} msec")

        print("\n取引明細")
        df_transaction = agent.getTransaction()
        print(df_transaction)
        print(f"一株当りの損益 : {df_transaction['損益'].sum()} 円")
