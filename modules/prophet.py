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
    __version__ = "0.0.3"
    __author__ = "Fuhito Suguri"
    __license__ = "MIT"

    requestResetEnv = Signal()
    requestParam = Signal()
    requestPostProcs = Signal()
    sendTradeData = Signal(float, float, float)

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)  # モジュール固有のロガーを取得
        self.res = res = AppRes()
        self.df = None
        self.row = 0
        self.t_start = 0

        # 実行用パラメータ（主にデータ、銘柄コードなど）
        self.dict_info = dict()

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
        print("\nループを終了しました。")
        print(f"計測時間 :\t\t{t_delta:,.3f} sec")
        print(f"ティック数 :\t\t{self.row - 1 :,d} ticks")
        print(f"単位処理時間 :\t{t_delta / (self.row - 1) * 1_000:.3f} msec")

        # 後処理をリクエスト
        self.requestPostProcs.emit()

    def on_debug(self):
        """
        機能確認用
        :return:
        """
        pass

    def on_start(self):
        """
        スタートボタンがクリックされた時の処理
        :return:
        """
        # 選択されたモデルと過去ティックデータ、銘柄コードを取得
        self.dict_info = dict_info = self.toolbar.getInfo()

        print("\n下記の条件で推論を実施します。")
        path_excel, code = self.get_file_code()
        print(f"ティックデータ\t: {path_excel}")
        print(f"銘柄コード\t: {code}")

        # Excel ファイルをデータフレームに読み込む
        self.df = get_excel_sheet(path_excel, code)
        print("\nExcel ファイルをデータフレームに読み込みました。")

        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # 推論用スレッドの開始
        print("\nワーカースレッドを生成・開始します。")
        self.start_thread()
        # 必要なパラメータをエージェント側から取得してティックデータのチャートを作成
        self.requestParam.emit()
        # エージェント環境のリセット → リセット終了で推論開始
        self.requestResetEnv.emit()
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_

    def get_file_code(self) -> tuple[str, str]:
        path_excel = self.dict_info["path_excel"]
        code = self.dict_info["code"]
        return path_excel, code

    def plot_chart(self, dict_param: dict):
        # ティックデータのプロット
        path_excel, code = self.get_file_code()
        title = f"{os.path.basename(path_excel)}, {code}"
        self.win_tick.draw(self.df, dict_param, title)

    def post_process(self, dict_result: dict):
        """
        ループ後の処理
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
        self.row = 0  # 行位置を 0 にリセット
        print("ループを開始します。")
        self.t_start = perf_counter()  # ループ開始時刻
        self.send_one_tick()  # 一行のデータをエージェントに送る

    def send_one_tick(self):
        """
        ひとつずつティックデータを送ってリアルタイムをシミュレート
        :return:
        """
        if self.row >= len(self.df):
            # データフレームの末尾で終了
            self.finished_trading()
        else:
            # データフレームからティックデータを１セット取得
            ts = float(self.df["Time"].iloc[self.row])
            price = float(self.df["Price"].iloc[self.row])
            volume = float(self.df["Volume"].iloc[self.row])
            # エージェントにティックデータを送る
            self.sendTradeData.emit(ts, price, volume)
            # 行位置をインクリメント
            self.row += 1

    def start_thread(self):
        """
        スレッドの開始
        :param path_model:
        :return:
        """
        self.thread = QThread(self)
        # self.worker = WorkerAgent(path_model, True)
        self.worker = WorkerAgent(True)  # モデルを使わないアルゴリズム取引用
        self.worker.moveToThread(self.thread)

        self.requestResetEnv.connect(self.worker.resetEnv)
        self.requestParam.connect(self.worker.getParam)
        self.requestPostProcs.connect(self.worker.postProcs)
        self.sendTradeData.connect(self.worker.addData)

        self.worker.completedResetEnv.connect(self.send_first_tick)
        self.worker.completedTrading.connect(self.finished_trading)
        self.worker.readyNext.connect(self.send_one_tick)
        self.worker.sendParam.connect(self.plot_chart)
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

        print("\nスレッドを終了しました。")
