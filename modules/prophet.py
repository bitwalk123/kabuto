import logging
import os
from time import perf_counter

import pandas as pd
from PySide6.QtCore import QThread, Signal
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

    requestReset = Signal()
    requestPostProcs = Signal()
    sendTradeData = Signal(float, float, float)

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)  # モジュール固有のロガーを取得
        self.res = res = AppRes()
        self.df = None
        self.done = False  # 推論終了フラグ
        self.row = 0
        self.t_start = 0

        self.setWindowIcon(QIcon(os.path.join(res.dir_image, "inference.png")))
        title_win = f"{self.__app_name__} - {self.__version__}"
        self.setWindowTitle(title_win)

        self.toolbar = toolbar = ToolBarProphet(res)
        toolbar.clickedPlay.connect(self.on_start)
        self.addToolBar(toolbar)

        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # 強化学習モデル用スレッド
        self.thread = None
        self.worker = None

    def conclude_result(self, dict_result: dict):
        print("\n取引明細")
        df_transaction = dict_result["transaction"]
        print(df_transaction)
        print(f"一株当りの損益 : {df_transaction['損益'].sum()} 円")

        # スレッドの終了
        self.stop_thread()

    def finished_trading(self):
        t_end = perf_counter()  # ループ終了時刻
        t_delta = t_end - self.t_start
        print("\n推論ループを終了しました。")
        print(f"計測時間 :\t\t\t{t_delta:,.3f} sec")
        print(f"ティック数 :\t\t\t{self.row - 1 :,d} ticks")
        print(f"処理時間 / ティック :\t{t_delta / (self.row - 1) * 1_000:.3f} msec")

    def on_start(self):
        dict_info = self.toolbar.getInfo()
        path_model: str = dict_info["path_model"]
        path_excel: str = dict_info["path_excel"]
        code: str = dict_info["code"]

        print("\n下記の条件で推論を実施します。")
        print(f"モデル\t\t: {path_model}")
        print(f"ティックデータ\t: {path_excel}")
        print(f"銘柄コード\t: {code}")

        # Excel ファイルをデータフレームに読み込む
        self.df = get_excel_sheet(path_excel, code)
        print("\nExcel ファイルをデータフレームに読み込みました。")
        print(self.df)

        # スレッドの開始
        print("\nスレッド内にワーカーエージェントを生成します。")
        self.start_thread(path_model)

        # エージェント環境のリセット
        self.requestReset.emit()
        self.row = 0
        print("\n推論ループを開始します。")
        self.t_start = perf_counter()  # ループ開始時刻
        self.simulation()

    def simulation(self):
        ts = float(self.df["Time"].iloc[self.row])
        price = float(self.df["Price"].iloc[self.row])
        volume = float(self.df["Volume"].iloc[self.row])
        self.sendTradeData.emit(ts, price, volume)
        self.row += 1
        if self.row > len(self.df):
            self.done = True

    def start_thread(self, path_model: str):
        """
        スレッドの開始
        :param path_model:
        :return:
        """
        self.thread = QThread(self)
        self.worker = WorkerAgent(path_model, True)
        self.worker.moveToThread(self.thread)

        self.requestReset.connect(self.worker.resetEnv)
        self.requestPostProcs.connect(self.worker.postProcs)

        self.sendTradeData.connect(self.worker.addData)
        self.worker.completedResetEnv.connect(self.simulation)
        self.worker.completedTrading.connect(self.finished_trading)
        self.worker.readyNext.connect(self.simulation)
        self.worker.sendResults.connect(self.conclude_result)
        self.thread.start()

    def stop_thread(self):
        """
        スレッド終了
        :return:
        """
        self.thread.quit()
        self.thread.wait()

        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None

        if self.thread is not None:
            self.thread.deleteLater()
            self.thread = None
