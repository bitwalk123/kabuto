import datetime
import gc
import logging
import os
import sys
from time import perf_counter

import pandas as pd
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QMainWindow, QApplication

from funcs.ios import get_excel_sheet
from funcs.tide import get_datetime_str
from modules.agent import WorkerAgent
from modules.win_obs import WinObs
from widgets.toolbars import ToolBarProphet
from structs.app_enum import AppMode
from modules.win_tick import WinTick
from structs.res import AppRes
from widgets.containers import TabWidget
from widgets.statusbars import StatusBar


class Prophet(QMainWindow):
    __app_name__ = "Prophet"
    __version__ = "0.0.7"
    __author__ = "Fuhito Suguri"
    __license__ = "MIT"

    requestObs = Signal()
    requestForceRepay = Signal()
    requestParams = Signal()
    requestPostProcs = Signal()
    requestResetEnv = Signal()
    sendTradeData = Signal(float, float, float)

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)  # モジュール固有のロガーを取得
        self.res = res = AppRes()
        self.df = None
        self.row = 0
        self.t_start = 0

        # 実行用パラメータ（主にデータ、銘柄コードなど）格納用
        self.dict_info = dict()
        # 環境内のテクニカル指標などのパラメータ格納用
        self.dict_param = dict()

        self.code = ''
        self.path_excel = ''
        """
        ALL, DOE モードで実行するティックデータのリスト
        """
        self.list_tick = []
        self.idx_tick = 0
        self.dict_all = {
            "file": [],
            "code": [],
            "trade": [],
            "total": [],
        }
        """
        DOE 用
        """
        # self.name_doe = "doe-1"
        # self.name_doe = "doe-2"
        # self.name_doe = "doe-3"
        # self.name_doe = "doe-4"
        # self.name_doe = "doe-5"
        self.name_doe = "doe-6"
        self.row_condition = 0
        self.dict_doe = dict()  # DOE 用
        """
        # doe-1
        self.factor_doe = ["PERIOD_MA_1", "PERIOD_MA_2", "PERIOD_MR", "THRESHOLD_MR"]
        self.df_matrix = pd.DataFrame({
            "PERIOD_MA_1": [
                30, 60, 90, 30, 60, 90, 30, 60, 90,
                30, 60, 90, 30, 60, 90, 30, 60, 90,
                30, 60, 90, 30, 60, 90, 30, 60, 90,
                30, 60, 90, 30, 60, 90, 30, 60, 90,
                30, 60, 90, 30, 60, 90, 30, 60, 90,
                30, 60, 90, 30, 60, 90, 30, 60, 90,
                30, 60, 90, 30, 60, 90, 30, 60, 90,
                30, 60, 90, 30, 60, 90, 30, 60, 90,
                30, 60, 90, 30, 60, 90, 30, 60, 90,
            ],
            "PERIOD_MA_2": [
                300, 300, 300, 600, 600, 600, 900, 900, 900,
                300, 300, 300, 600, 600, 600, 900, 900, 900,
                300, 300, 300, 600, 600, 600, 900, 900, 900,
                300, 300, 300, 600, 600, 600, 900, 900, 900,
                300, 300, 300, 600, 600, 600, 900, 900, 900,
                300, 300, 300, 600, 600, 600, 900, 900, 900,
                300, 300, 300, 600, 600, 600, 900, 900, 900,
                300, 300, 300, 600, 600, 600, 900, 900, 900,
                300, 300, 300, 600, 600, 600, 900, 900, 900,
            ],
            "PERIOD_MR": [
                15, 15, 15, 15, 15, 15, 15, 15, 15,
                30, 30, 30, 30, 30, 30, 30, 30, 30,
                45, 45, 45, 45, 45, 45, 45, 45, 45,
                15, 15, 15, 15, 15, 15, 15, 15, 15,
                30, 30, 30, 30, 30, 30, 30, 30, 30,
                45, 45, 45, 45, 45, 45, 45, 45, 45,
                15, 15, 15, 15, 15, 15, 15, 15, 15,
                30, 30, 30, 30, 30, 30, 30, 30, 30,
                45, 45, 45, 45, 45, 45, 45, 45, 45,
            ],
            "THRESHOLD_MR": [
                1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1,
                4, 4, 4, 4, 4, 4, 4, 4, 4,
                4, 4, 4, 4, 4, 4, 4, 4, 4,
                4, 4, 4, 4, 4, 4, 4, 4, 4,
                7, 7, 7, 7, 7, 7, 7, 7, 7,
                7, 7, 7, 7, 7, 7, 7, 7, 7,
                7, 7, 7, 7, 7, 7, 7, 7, 7,
            ],
        })
        # doe-2
        self.factor_doe = ["PERIOD_MA_1", "PERIOD_MA_2", "THRESHOLD_MR"]
        self.df_matrix = pd.DataFrame({
            "PERIOD_MA_1": [
                60, 90, 120, 60, 90, 120, 60, 90, 120,
                60, 90, 120, 60, 90, 120, 60, 90, 120,
                60, 90, 120, 60, 90, 120, 60, 90, 120,
            ],
            "PERIOD_MA_2": [
                300, 300, 300, 600, 600, 600, 900, 900, 900,
                300, 300, 300, 600, 600, 600, 900, 900, 900,
                300, 300, 300, 600, 600, 600, 900, 900, 900,
            ],
            "THRESHOLD_MR": [
                3, 3, 3, 3, 3, 3, 3, 3, 3,
                5, 5, 5, 5, 5, 5, 5, 5, 5,
                7, 7, 7, 7, 7, 7, 7, 7, 7,
            ],
        })
        # doe-3
        self.factor_doe = ["PERIOD_MA_1", "PERIOD_MA_2"]
        self.df_matrix = pd.DataFrame({
            "PERIOD_MA_1": [
                50, 70, 90, 110, 130,
                50, 70, 90, 110, 130,
                50, 70, 90, 110, 130,
                50, 70, 90, 110, 130,
                50, 70, 90, 110, 130,
            ],
            "PERIOD_MA_2": [
                200, 200, 200, 200, 200,
                400, 400, 400, 400, 400,
                600, 600, 600, 600, 600,
                800, 800, 800, 800, 800,
                1000, 1000, 1000, 1000, 1000,
            ],
        })
        # doe-4
        self.factor_doe = ["PERIOD_MA_1", "PERIOD_MA_2"]
        self.df_matrix = pd.DataFrame({
            "PERIOD_MA_1": [
                60, 90, 120, 150, 180,
                60, 90, 120, 150, 180,
                60, 90, 120, 150, 180,
                60, 90, 120, 150, 180,
                60, 90, 120, 150, 180,
            ],
            "PERIOD_MA_2": [
                500, 500, 500, 500, 500,
                600, 600, 600, 600, 600,
                700, 700, 700, 700, 700,
                800, 800, 800, 800, 800,
                900, 900, 900, 900, 900,
            ],
        })
        # doe-5
        self.factor_doe = ["PERIOD_MA_1", "PERIOD_MA_2"]
        self.df_matrix = pd.DataFrame({
            "PERIOD_MA_1": [
                30, 60, 90, 120, 150,
                30, 60, 90, 120, 150,
                30, 60, 90, 120, 150,
                30, 60, 90, 120, 150,
                30, 60, 90, 120, 150,
            ],
            "PERIOD_MA_2": [
                300, 300, 300, 300, 300,
                400, 400, 400, 400, 400,
                500, 500, 500, 500, 500,
                600, 600, 600, 600, 600,
                700, 700, 700, 700, 700,
            ],
        })
        """
        # doe-6
        self.factor_doe = ["PERIOD_MA_1", "PERIOD_MA_2"]
        self.df_matrix = pd.DataFrame({
            "PERIOD_MA_1": [
                30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180,
                30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180,
                30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180,
                30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180,
                30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180,
                30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180,
                30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180,
                30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180,
                30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180,
                30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180,
                30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180,
            ],
            "PERIOD_MA_2": [
                300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300,
                360, 360, 360, 360, 360, 360, 360, 360, 360, 360, 360,
                420, 420, 420, 420, 420, 420, 420, 420, 420, 420, 420,
                480, 480, 480, 480, 480, 480, 480, 480, 480, 480, 480,
                540, 540, 540, 540, 540, 540, 540, 540, 540, 540, 540,
                600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600,
                660, 660, 660, 660, 660, 660, 660, 660, 660, 660, 660,
                720, 720, 720, 720, 720, 720, 720, 720, 720, 720, 720,
                780, 780, 780, 780, 780, 780, 780, 780, 780, 780, 780,
                840, 840, 840, 840, 840, 840, 840, 840, 840, 840, 840,
                900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900,
            ],
        })

        # 強化学習モデル用スレッド
        self.thread = None
        self.worker = None

        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # GUI
        self.setWindowIcon(QIcon(os.path.join(res.dir_image, "inference.png")))
        title_win = f"{self.__app_name__} - {self.__version__}"
        self.setWindowTitle(title_win)

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
        self.win_obj = win_obs = WinObs(res)
        tabbase.addTab(win_obs, "観測値チャート")

    def finished_trading(self):
        # ベンチマーク計測
        t_end = perf_counter()  # ループ終了時刻
        t_delta = t_end - self.t_start
        # ベンチマーク出力
        print("ループを終了しました。")
        print(f"計測時間 :\t\t{t_delta:,.3f} sec")
        print(f"ティック数 :\t\t{self.row - 1 :,d} ticks")
        print(f"単位処理時間 :\t{t_delta / (self.row - 1) * 1_000:.3f} msec")
        # 後処理をリクエスト
        self.requestPostProcs.emit()
        # 処理した観測値をリクエスト
        self.requestObs.emit()

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
        self.dict_info = self.toolbar.getInfo()
        mode = self.dict_info["mode"]
        if mode == AppMode.SINGLE:
            self.start_mode_single()
        elif mode == AppMode.ALL:
            self.list_tick = self.toolbar.getListTicks(reverse=False)
            self.idx_tick = 0
            self.start_mode_all()
        elif mode == AppMode.DOE:
            """
            ■■■ DOE 用 ■■■
            ティックファイル・リスト
            """
            self.list_tick = self.toolbar.getListTicks(reverse=False)[-23:-18]
            """
            self.list_tick = []
            """
            self.idx_tick = 0
            self.start_mode_doe()
        else:
            raise TypeError(f"Unknown AppMode: {mode}")

    def get_file_code_all(self) -> tuple[str, str]:
        path_excel = str(
            os.path.join(
                self.res.dir_collection,
                self.list_tick[self.idx_tick]
            )
        )
        code = self.dict_info["code"]
        return path_excel, code

    def get_file_code_doe(self) -> tuple[str, str]:
        path_excel = str(
            os.path.join(
                self.res.dir_collection,
                self.list_tick[self.idx_tick]
            )
        )
        code = self.dict_info["code"]
        return path_excel, code

    def get_file_code_single(self) -> tuple[str, str]:
        path_excel = self.dict_info["path_excel"]
        code = self.dict_info["code"]
        return path_excel, code

    def get_file_output(self, file_excel) -> str:
        file_body_without_ext = os.path.splitext(os.path.basename(file_excel))[0]
        path_result = os.path.join(
            self.res.dir_output,
            self.name_doe,
            self.code,
            f"{file_body_without_ext}.csv"
        )
        return path_result

    def plot_obs(self, df_obs: pd.DataFrame):
        df_obs.index = pd.to_datetime(
            [datetime.datetime.fromtimestamp(ts) for ts in df_obs["Timestamp"]]
        )
        title = f"{os.path.basename(self.path_excel)}, {self.code}"
        cols = [l for l in df_obs.columns if l not in ["Timestamp", "Volume"]]
        self.win_obj.draw(df_obs[cols], self.dict_param, title)

    def plot_tick(self, dict_param: dict):
        self.dict_param = dict_param
        # ティックデータのプロット
        title = f"{os.path.basename(self.path_excel)}, {self.code}"
        self.win_tick.draw(self.df, dict_param, title)

    def post_procs(self, dict_result: dict):
        """
        ループ後の処理
        :param dict_result:
        :return:
        """
        # 行の表示数の上限
        pd.set_option('display.max_rows', 100)
        print("\n【取引明細】")
        df_transaction: pd.DataFrame = dict_result["transaction"]
        print(df_transaction)
        n_trade = len(df_transaction)
        total = df_transaction['損益'].sum()
        print(f"取引回数 : {n_trade} 回, 一株当りの損益 : {total} 円")

        mode = self.dict_info["mode"]
        if mode == AppMode.ALL:
            self.dict_all["file"].append(os.path.basename(self.path_excel))
            self.dict_all["code"].append(self.code)
            self.dict_all["trade"].append(n_trade)
            self.dict_all["total"].append(total)
        if mode == AppMode.DOE:
            # file
            key = "file"
            value = os.path.basename(self.path_excel)
            self.dict_doe.setdefault(key, []).append(value)
            # code
            key = "code"
            value = self.code
            self.dict_doe.setdefault(key, []).append(value)
            # trade
            key = "trade"
            value = n_trade
            self.dict_doe.setdefault(key, []).append(value)

            # Experiment Factors
            for key in self.factor_doe:
                value = self.df_matrix.at[self.row_condition, key]
                self.dict_doe.setdefault(key, []).append(value)
            '''
            # PERIOD_MA_2
            key = "PERIOD_MA_2"
            value = self.df_matrix.at[self.row_condition, key]
            self.dict_doe.setdefault(key, []).append(value)
            # PERIOD_MR
            key = "PERIOD_MR"
            value = self.df_matrix.at[self.row_condition, key]
            self.dict_doe.setdefault(key, []).append(value)
            # THRESHOLD_MR
            key = "THRESHOLD_MR"
            value = self.df_matrix.at[self.row_condition, key]
            self.dict_doe.setdefault(key, []).append(value)
            '''

            # 4. total
            key = "total"
            value = total
            self.dict_doe.setdefault(key, []).append(value)

        # スレッドの終了
        self.stop_thread()

    def post_procs_all(self):
        print("ALL モードの全ループを終了しました。")
        df_all = pd.DataFrame(self.dict_all)
        print(df_all)
        print(f"合計: {df_all['total'].sum(): ,.0f}")
        datetime_str = get_datetime_str()
        path_result = os.path.join(self.res.dir_log, f"result_{datetime_str}.csv")
        print(f"結果を {path_result} へ保存しました。")
        df_all.to_csv(path_result, index=False)

    def post_procs_doe(self):
        print("DOE モード")
        file_excel = self.list_tick[self.idx_tick]
        print(f"{file_excel} / {self.code} で DOE ループを終了しました。")
        df_doe = pd.DataFrame(self.dict_doe)
        print(df_doe)
        path_result = self.get_file_output(file_excel)
        # 　ディレクトリが存在していなかったら作成
        path_dir = os.path.dirname(path_result)
        if not os.path.isdir(path_dir):
            os.makedirs(path_dir)
        df_doe.to_csv(path_result, index=False)
        print(f"結果を {path_result} へ保存しました。")

        if len(self.list_tick) - 1 <= self.idx_tick:
            print(f"対象のティックファイル全てで全ループを終了しました。")
        else:
            self.idx_tick += 1
            self.row_condition = 0
            self.dict_doe = dict()
            # 新たなティックファイルで DOE ループ
            self.start_mode_doe()

    def send_first_tick(self):
        """
        環境をリセットした後の最初のティックデータ送信
        :return:
        """
        print("環境がリセットされました。")
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
            # 建玉があれば強制返済
            self.requestForceRepay.emit()
            # データフレームの末尾で終了
            print("ティックデータの末尾に達しました。")
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

    def start_mode_all(self):
        """
        過去データ全て
        :return:
        """
        self.path_excel, self.code = self.get_file_code_all()
        self.idx_tick += 1
        print(f"ティックデータ\t: {self.path_excel}")
        print(f"銘柄コード\t: {self.code}")

        # Excel ファイルをデータフレームに読み込む
        self.df = get_excel_sheet(self.path_excel, self.code)
        print("Excel ファイルをデータフレームに読み込みました。")

        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # スレッドの開始
        self.start_thread(self.code, dict())
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_

    def start_mode_doe(self):
        """
        DOE モード
        :return:
        """
        self.path_excel, self.code = self.get_file_code_doe()
        print(f"ティックデータ\t: {self.path_excel}")
        print(f"銘柄コード\t: {self.code}")

        # Excel ファイルをデータフレームに読み込む
        self.df = get_excel_sheet(self.path_excel, self.code)
        print("Excel ファイルをデータフレームに読み込みました。")
        dict_param = dict()
        for key in self.factor_doe:
            dict_param[key] = int(self.df_matrix.at[self.row_condition, key])

        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # スレッドの開始
        self.start_thread(self.code, dict_param)
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_

    def start_mode_single(self):
        print("下記の条件で取引シミュレーションを実施します。")
        self.path_excel, self.code = self.get_file_code_single()
        print(f"ティックデータ\t: {self.path_excel}")
        print(f"銘柄コード\t: {self.code}")

        # Excel ファイルをデータフレームに読み込む
        self.df = get_excel_sheet(self.path_excel, self.code)
        print("Excel ファイルをデータフレームに読み込みました。")

        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # スレッドの開始
        self.start_thread(self.code, dict())
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_

    def start_thread(self, code: str, dict_param: dict):
        """
        スレッドの開始
        :param:
        :return:
        """
        print("ワーカースレッドを生成・開始します。")
        self.thread = QThread(self)
        # self.worker = WorkerAgent(path_model, True)
        self.worker = WorkerAgent(True, code, dict_param)  # モデルを使わないアルゴリズム取引用
        self.worker.moveToThread(self.thread)

        self.requestForceRepay.connect(self.worker.forceRepay)
        self.requestObs.connect(self.worker.getObs)
        self.requestParams.connect(self.worker.getParams)
        self.requestPostProcs.connect(self.worker.postProcs)
        self.requestResetEnv.connect(self.worker.resetEnv)
        self.sendTradeData.connect(self.worker.addData)

        self.worker.completedResetEnv.connect(self.send_first_tick)
        self.worker.completedTrading.connect(self.finished_trading)
        self.worker.readyNext.connect(self.send_one_tick)
        self.worker.sendObs.connect(self.plot_obs)
        self.worker.sendParams.connect(self.plot_tick)
        self.worker.sendResults.connect(self.post_procs)

        self.thread.start()

        # 必要なパラメータをエージェント側から取得してティックデータのチャートを作成
        self.requestParams.emit()
        # エージェント環境のリセット → リセット終了で処理開始
        self.requestResetEnv.emit()

    def stop_thread(self):
        """
        スレッド終了
        :return:
        """
        self.requestResetEnv.disconnect()
        self.requestPostProcs.disconnect()
        self.sendTradeData.disconnect()

        if self.thread is not None:
            self.thread.quit()
            self.thread.wait()

        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None

        if self.thread is not None:
            self.thread.deleteLater()
            self.thread = None

        print("\nスレッドを終了しました。")
        QApplication.processEvents()  # Qt の deleteLater を実行させる
        gc.collect()  # Python 側の孤立オブジェクトを回収

        mode = self.dict_info["mode"]
        if mode == AppMode.SINGLE:
            print("SINGLE モードを終了しました。")
        elif mode == AppMode.ALL:
            if len(self.list_tick) <= self.idx_tick:
                self.post_procs_all()
            else:
                print("次のティックデータに進みます（ALL モード）。")
                self.start_mode_all()
        elif mode == AppMode.DOE:
            if len(self.df_matrix) - 1 <= self.row_condition:
                self.post_procs_doe()
            else:
                print("次の条件に進みます（DOE モード）。")
                self.row_condition += 1
                self.start_mode_doe()
        else:
            raise TypeError(f"Unknown AppMode: {mode}")
