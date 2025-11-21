import logging
import os
from time import perf_counter

import pandas as pd
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QMainWindow

from funcs.ios import get_excel_sheet
from modules.agent import WorkerAgent
from modules.win_tick import WinTick
from widgets.statusbars import StatusBar
from modules.toolbar import ToolBarProphet
from structs.res import AppRes
from widgets.containers import TabWidget


class Prophet(QMainWindow):
    __app_name__ = "Prophet"
    __version__ = "0.0.2"
    __author__ = "Fuhito Suguri"
    __license__ = "MIT"

    requestResetEnv = Signal()
    requestPostProcs = Signal()
    sendTradeData = Signal(float, float, float)

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)  # モジュール固有のロガーを取得
        self.res = res = AppRes()
        self.df = None
        self.row = 0
        self.t_start = 0

        # 強化学習モデル用スレッド
        self.thread = None
        self.worker = None

        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # GUI
        self.setWindowIcon(QIcon(os.path.join(res.dir_image, "inference.png")))
        title_win = f"{self.__app_name__} - {self.__version__}"
        self.setWindowTitle(title_win)
        self.setFixedSize(1500, 500)

        # =====================================================================
        # ツールバー
        # =====================================================================
        self.toolbar = toolbar = ToolBarProphet(res)
        toolbar.clickedDebug.connect(self.on_debug)
        toolbar.clickedPlay.connect(self.on_start)
        self.addToolBar(toolbar)

        # =====================================================================
        # ステータスバー
        # =====================================================================
        self.statusbar = statusbar = StatusBar(res)
        self.setStatusBar(statusbar)

        # =====================================================================
        # メイン・ウィンドウ
        # =====================================================================
        self.tab_base = tabbase = TabWidget()
        self.tab_base.setTabPosition(TabWidget.TabPosition.South)
        self.setCentralWidget(tabbase)
        # ---------------------------------------------------------------------
        # タブオブジェクト
        # ---------------------------------------------------------------------
        self.win_tick = win_tick = WinTick(res)
        tabbase.addTab(win_tick, "ティックチャート")

    def finished_trading(self):
        t_end = perf_counter()  # ループ終了時刻
        t_delta = t_end - self.t_start
        print("\n推論ループを終了しました。")
        print(f"計測時間 :\t\t\t{t_delta:,.3f} sec")
        print(f"ティック数 :\t\t\t{self.row - 1 :,d} ticks")
        print(f"処理時間 / ティック :\t{t_delta / (self.row - 1) * 1_000:.3f} msec")

        # 後処理をリクエスト
        self.requestPostProcs.emit()

    def on_debug(self):
        dict_info = self.toolbar.getInfo()
        path_excel: str = dict_info["path_excel"]
        code: str = dict_info["code"]
        df = get_excel_sheet(path_excel, code)

        title = f"{os.path.basename(path_excel)}, {code}"
        self.win_tick.draw(df, title)

    def on_start(self):
        """
        スタートボタンがクリックされた時の処理
        :return:
        """
        # 選択されたモデルと過去ティックデータ、銘柄コードを取得
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

        # ティックデータのプロット
        title = f"{os.path.basename(path_excel)}, {code}"
        self.win_tick.draw(self.df, title)

        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # 推論用スレッドの開始
        print("\nスレッド内にワーカーエージェントを生成します。")
        self.start_thread(path_model)
        # エージェント環境のリセット → リセット終了で推論開始
        self.requestResetEnv.emit()
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_


    def post_process(self, dict_result: dict):
        """
        推論後の処理
        :param dict_result:
        :return:
        """
        print("\n取引明細")
        df_transaction: pd.DataFrame = dict_result["transaction"]
        print(df_transaction)
        print(f"一株当りの損益 : {df_transaction['損益'].sum()} 円")

        # スレッドの終了
        self.stop_thread()

    def send_first_tick(self):
        """
        環境をリセットした後の最初のティックデータ送信
        :return:
        """
        print("\n環境がリセットされました。")
        self.row = 0
        print("推論ループを開始します。")
        self.t_start = perf_counter()  # ループ開始時刻
        self.send_one_tick()

    def send_one_tick(self):
        """
        ひとつずつティックデータを送ってリアルタイムをシミュレート
        :return:
        """
        # データフレームからティックデータを１セット取得
        ts = float(self.df["Time"].iloc[self.row])
        price = float(self.df["Price"].iloc[self.row])
        volume = float(self.df["Volume"].iloc[self.row])
        # エージェントにティックデータを送る
        self.sendTradeData.emit(ts, price, volume)
        # 行位置をインクリメント
        self.row += 1
        # 行位置が最後であれば終了（最後の行は使わない）
        if self.row >= len(self.df):
            self.finished_trading()

    def start_thread(self, path_model: str):
        """
        スレッドの開始
        :param path_model:
        :return:
        """
        self.thread = QThread(self)
        self.worker = WorkerAgent(path_model, True)
        self.worker.moveToThread(self.thread)

        self.requestResetEnv.connect(self.worker.resetEnv)
        self.requestPostProcs.connect(self.worker.postProcs)
        self.sendTradeData.connect(self.worker.addData)

        self.worker.completedResetEnv.connect(self.send_first_tick)
        self.worker.completedTrading.connect(self.finished_trading)
        self.worker.readyNext.connect(self.send_one_tick)
        self.worker.sendResults.connect(self.post_process)

        self.thread.start()

    def stop_thread(self):
        """
        スレッド終了
        :return:
        """
        self.requestResetEnv.disconnect()
        self.requestPostProcs.disconnect()
        self.sendTradeData.disconnect()

        self.thread.quit()
        self.thread.wait()

        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None

        if self.thread is not None:
            self.thread.deleteLater()
            self.thread = None
