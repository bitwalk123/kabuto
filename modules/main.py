import logging
import os
import time

import pandas as pd
from PySide6.QtCore import (
    QThread,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QCloseEvent,
    QIcon,
)
from PySide6.QtWidgets import QMainWindow

from funcs.ios import save_dataframe_to_excel
from funcs.tide import get_intraday_timestamp
from funcs.uis import clear_boxlayout
from modules.dialog import DlgAboutThis
from modules.dock import DockTrader
from modules.reviewer import ExcelReviewWorker
from modules.rssreader import RSSReaderWorker
from modules.statusbar import StatusBar
from modules.toolbar import ToolBar
from modules.trader import Trader
from modules.trans import WinTransaction
from structs.app_enum import ActionType
from structs.res import AppRes
from widgets.containers import Widget
from widgets.layouts import VBoxLayout


class Kabuto(QMainWindow):
    __app_name__ = "Kabuto"
    __version__ = "0.11.0"
    __author__ = "Fuhito Suguri"
    __license__ = "MIT"

    # ワーカーの初期化シグナル
    requestWorkerInit = Signal()
    # 現在価格取得リクエスト・シグナル
    requestCurrentPrice = Signal(float)
    requestSaveDataFrame = Signal()
    requestStopProcess = Signal()

    # 売買
    requestPositionOpen = Signal(str, float, float, ActionType, str)
    requestPositionClose = Signal(str, float, float, str)
    requestTransactionResult = Signal()

    # このスレッドが開始されたことを通知するシグナル（デバッグ用など）
    threadReady = Signal()

    def __init__(self, excel_path: str, debug: bool = True):
        super().__init__()
        self.logger = logging.getLogger(__name__)  # モジュール固有のロガーを取得
        self.res = res = AppRes()
        res.debug = debug  # デバッグ・モードを保持

        #######################################################################
        # NORMAL / DEBUG モード固有の設定
        if debug:
            # DEBUG モード
            self.logger.info(f"{__name__}: executed as DEBUG mode!")
            self.timer_interval = 100  # タイマー間隔（ミリ秒）（デバッグ時）
            self.flag_data_ready = False
        else:
            # NORMAL モード
            self.logger.info(f"{__name__}: executed as NORMAL mode!")
            self.timer_interval = 1000  # タイマー間隔（ミリ秒）
        #
        #######################################################################

        # ---------------------------------------------------------------------
        # 株価取得スレッド用インスタンス
        # ---------------------------------------------------------------------
        self.thread = QThread(self)
        self.worker = None

        # ---------------------------------------------------------------------
        # Trader インスタンス
        # 銘柄コード別にチャートや売買情報および売買機能の UI を提供する
        # ---------------------------------------------------------------------
        self.trader: Trader | None = None
        # インスタンスを保持する辞書
        self.dict_trader = dict()

        # ---------------------------------------------------------------------
        # 取引履歴
        # ---------------------------------------------------------------------
        self.df_transaction = None
        self.win_transaction: WinTransaction | None = None

        # システム時刻（タイムスタンプ形式）
        self.ts_system = 0

        # ザラ場の開始時間などのタイムスタンプ取得（本日分）
        self.dict_ts = get_intraday_timestamp()

        # 取引が終了したかどうかのフラグ
        self.finished_trading = False

        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        #  UI
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_

        # ウィンドウアイコンとタイトルを設定
        self.setWindowIcon(QIcon(os.path.join(res.dir_image, "beetle.png")))
        title_win = f"{self.__app_name__} - {self.__version__}"
        if debug:
            # デバッグモードを示す文字列を追加
            title_win = f"{title_win} [debug mode]"
        self.setWindowTitle(title_win)

        # ---------------------------------------------------------------------
        # ツールバー
        # ---------------------------------------------------------------------
        self.toolbar = toolbar = ToolBar(res)
        toolbar.clickedAbout.connect(self.on_about)
        toolbar.clickedPlay.connect(self.on_review_play)
        toolbar.clickedStop.connect(self.on_review_stop)
        toolbar.clickedTransaction.connect(self.on_show_transaction)
        toolbar.selectedExcelFile.connect(self.on_create_thread_review)
        self.addToolBar(toolbar)

        # ---------------------------------------------------------------------
        # ステータスバー
        # ---------------------------------------------------------------------
        self.statusbar = statusbar = StatusBar(res)
        self.setStatusBar(statusbar)

        # ---------------------------------------------------------------------
        # メイン・ウィジェット
        # ---------------------------------------------------------------------
        base = Widget()
        self.setCentralWidget(base)
        self.layout = layout = VBoxLayout()
        base.setLayout(layout)

        # ---------------------------------------------------------------------
        # タイマー
        # ---------------------------------------------------------------------
        self.timer = timer = QTimer()
        timer.setInterval(self.timer_interval)

        if debug:
            # デバッグモードではファイルを読み込んでからスレッドを起動
            timer.timeout.connect(self.on_request_data_review)
        else:
            # リアルタイムモードでは、直ちにスレッドを起動
            timer.timeout.connect(self.on_request_data)
            # RSS用Excelファイルを指定してxlwingsを利用するスレッド
            self.on_create_thread(excel_path)

    def closeEvent(self, event: QCloseEvent):
        """
        アプリ終了イベント
        :param event:
        :return:
        """
        # ---------------------------------------------------------------------
        # タイマーの停止
        # ---------------------------------------------------------------------
        if self.timer.isActive():
            self.timer.stop()
            self.logger.info(f"{__name__}: timer stopped.")

        # ---------------------------------------------------------------------
        # self.thread スレッドの削除
        # ---------------------------------------------------------------------
        try:
            if self.thread.isRunning():
                self.requestStopProcess.emit()
                time.sleep(1)
                if self.worker:
                    self.worker.stop()
                    self.logger.info(f"{__name__}: deleted self.worker.")
                if self.thread:
                    self.thread.quit()
                    self.thread.wait()
                    self.logger.info(f"{__name__}: deleted self.thread.")
        except RuntimeError as e:
            self.logger.error(f"{__name__}: error at termination: {e}")

        # ---------------------------------------------------------------------
        self.logger.info(f"{__name__} stopped and closed.")
        event.accept()

    def create_trader(self, list_code: list, dict_name: dict, dict_lastclose: dict):
        """
        銘柄数分の Trader インスタンスの生成
        （リアルタイム・モード、デバッグ・モード共通）
        :param list_code:
        :param dict_name:
        :param dict_lastclose:
        :return:
        """
        # 配置済みの Trader インスタンスを消去
        clear_boxlayout(self.layout)
        # Trader 辞書のクリア
        self.dict_trader = dict()

        # 銘柄数分の Trader および Ticker インスタンスの生成
        for code in list_code:
            # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
            # Trader インスタンスの生成
            # 主にチャート表示用
            # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
            self.trader = trader = Trader(self.res, code)
            # Dock の売買ボタンのクリック・シグナルを直接ハンドリング
            trader.dock.clickedBuy.connect(self.on_buy)
            trader.dock.clickedRepay.connect(self.on_repay)
            trader.dock.clickedSell.connect(self.on_sell)

            # Trader 辞書に保持
            self.dict_trader[code] = trader

            # 「銘柄名　(code)」をタイトルにして設定し直し
            trader.setChartTitle(f"{dict_name[code]} ({code})")

            # 当日ザラ場時間
            trader.setTimeAxisRange(self.dict_ts["start"], self.dict_ts["end"])

            # 前日終値
            # if dict_lastclose[code] > 0:
            #    trader.setLastCloseLine(dict_lastclose[code])

            # 配置
            self.layout.addWidget(trader)

    def force_closing_position(self):
        self.logger.info(f"{__name__} 売買を強制終了します。")
        for code in self.dict_trader.keys():
            trader: Trader = self.dict_trader[code]
            dock: DockTrader = trader.dock
            dock.forceStopAutoPilot()

    def get_current_tick_data(self) -> dict:
        """
        チャートが保持しているティックデータをデータフレームで取得
        :return:
        """
        dict_df = dict()
        for code in self.dict_trader.keys():
            trader = self.dict_trader[code]
            dict_df[code] = trader.getTimePrice()
        return dict_df

    def on_about(self):
        """
        このアプリについて（ダイアログ表示）
        :return:
        """
        DlgAboutThis(
            self.res,
            self.__app_name__,
            self.__version__,
            self.__author__,
            self.__license__,
            "kabuto.png",
        ).exec()

    def on_create_thread(self, excel_path: str):
        """
        リアルタイム用ティックデータ取得スレッドの生成
        :param excel_path:
        :return:
        """
        # ---------------------------------------------------------------------
        # 00. リアルタイム用データ取得インスタンスの生成
        self.worker = RSSReaderWorker(excel_path)
        self.worker.moveToThread(self.thread)
        # ---------------------------------------------------------------------
        # 01. データ読み込み済みの通知（レビュー用のみ）
        # （なし）
        # =====================================================================
        # 02. スレッドが開始されたら、ワーカースレッド内で初期化処理を実行するシグナルを発行
        self.thread.started.connect(self.requestWorkerInit.emit)
        # ---------------------------------------------------------------------
        # 03. 初期化処理は主に xlwings 関連処理
        self.requestWorkerInit.connect(self.worker.initWorker)
        # ---------------------------------------------------------------------
        # 04. 売買ポジション処理用のメソッドへキューイング
        self.requestPositionOpen.connect(self.worker.posman.openPosition)
        self.requestPositionClose.connect(self.worker.posman.closePosition)
        # ---------------------------------------------------------------------
        # 05. 取引結果を取得するメソッドへキューイング
        self.requestTransactionResult.connect(self.worker.getTransactionResult)
        # ---------------------------------------------------------------------
        # 06. 現在株価を取得するメソッドへキューイング。
        self.requestCurrentPrice.connect(self.worker.readCurrentPrice)
        # ---------------------------------------------------------------------
        # 07. スレッドを終了する下記のメソッドへキューイング（リアルタイムでは xlwings 関連）。
        self.requestStopProcess.connect(self.worker.stopProcess)
        # =====================================================================
        # 08. 初期化後の銘柄情報を通知
        self.worker.notifyTickerN.connect(self.on_create_trader)
        # ---------------------------------------------------------------------
        # 09. タイマーで現在時刻と株価を通知
        self.worker.notifyCurrentPrice.connect(self.on_update_data)
        # ---------------------------------------------------------------------
        # 10. 取引結果を通知
        self.worker.notifyTransactionResult.connect(self.on_transaction_result)
        # ---------------------------------------------------------------------
        # 11. スレッド終了関連
        self.worker.threadFinished.connect(self.on_thread_finished)
        # =====================================================================
        # 12. スレッドを開始
        self.thread.start()

    def on_create_trader(self, list_code: list, dict_name: dict, dict_lastclose: dict):
        """
        Trader インスタンスの生成（リアルタイム）
        :param list_code:
        :param dict_name:
        :param dict_lastclose:
        :return:
        """
        # ---------------------------------------------------------------------
        # 銘柄数分の Trader インスタンスの生成
        # ---------------------------------------------------------------------
        self.create_trader(list_code, dict_name, dict_lastclose)

        if self.res.debug:
            # -----------------------------------------------------------------
            # デバッグの場合はスタート・ボタンがクリックされるまでは待機
            # -----------------------------------------------------------------
            self.logger.info(f"{__name__}: ready to review!")
        else:
            # -----------------------------------------------------------------
            # リアルタイムの場合はここでタイマーを開始
            # -----------------------------------------------------------------
            self.timer.start()
            self.logger.info(f"{__name__}: timer started!")

    def on_request_data(self):
        """
        タイマー処理（リアルタイム）
        """
        # システム時刻
        self.ts_system = time.time()
        if self.dict_ts["start"] <= self.ts_system <= self.dict_ts["end_1h"]:
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # 🧿 現在価格の取得要求をワーカースレッドに通知
            self.requestCurrentPrice.emit(self.ts_system)
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        elif self.dict_ts["start_2h"] <= self.ts_system <= self.dict_ts["end_2h"]:
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # 🧿 現在価格の取得要求をワーカースレッドに通知
            self.requestCurrentPrice.emit(self.ts_system)
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        elif self.dict_ts["end_2h"] < self.ts_system <= self.dict_ts["ca"]:
            if not self.finished_trading:
                # ポジションがあればクローズする
                self.force_closing_position()
                self.finished_trading = True
        elif self.dict_ts["ca"] < self.ts_system:
            self.timer.stop()
            self.logger.info(f"{__name__}: timer stopped!")
            # ティックデータの保存
            self.save_regular_tick_data()
            # 取引結果を取得
            self.requestTransactionResult.emit()
        else:
            pass

        # ツールバーの時刻を更新
        self.toolbar.updateTime(self.ts_system)

    def on_show_transaction(self):
        self.win_transaction = WinTransaction(self.res, self.df_transaction)
        self.win_transaction.show()

    def on_thread_finished(self, result: bool):
        """
        スレッド終了時のログ
        :param result:
        :return:
        """
        if result:
            self.logger.info(f"{__name__}: thread stopped normally.")
        else:
            self.logger.error(f"{__name__}: thread stopped abnormally.")

        if self.timer.isActive():
            self.timer.stop()
            self.logger.info(f"{__name__}: timer stopped")

    def on_ticker_ready(self, code: str):
        self.logger.info(f"{__name__}: thread for {code} is ready.")

    def on_transaction_result(self, df: pd.DataFrame):
        """
        取引結果のデータフレームを取得（リアルタイム、デバッグ・モード共通）
        :param df:
        :return:
        """
        print(df)
        print("合計損益", df["損益"].sum())

        # インスタンス変数に保存
        self.df_transaction = df

        # ツールバーの「取引履歴」ボタンを Enabled にする
        self.toolbar.set_transaction()

    def on_update_data(self, dict_data: dict, dict_profit: dict, dict_total: dict):
        """
        ティックデータ、含み益、損益の更新
        :param dict_data:
        :param dict_profit:
        :param dict_total:
        :return:
        """
        for code in dict_data.keys():
            x, y, vol = dict_data[code]
            trader: Trader = self.dict_trader[code]
            trader.setTradeData(x, y, vol)

            # 銘柄単位の現在株価および含み益と収益を更新
            trader.dock.setPrice(y)
            trader.dock.setProfit(dict_profit[code])
            trader.dock.setTotal(dict_total[code])

    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
    # 取引ボタンがクリックされた時の処理（Acquire 用）
    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
    def on_buy(self, code: str, price: float, note: str):
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 買建で建玉取得リクエストのシグナル
        self.requestPositionOpen.emit(
            code, self.ts_system, price, ActionType.BUY, note
        )
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def on_sell(self, code: str, price: float, note: str):
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 売建で建玉取得リクエストのシグナル
        self.requestPositionOpen.emit(
            code, self.ts_system, price, ActionType.SELL, note
        )
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def on_repay(self, code: str, price: float, note: str):
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 建玉返済リクエストのシグナル
        self.requestPositionClose.emit(
            code, self.ts_system, price, note
        )
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
    # ティックデータの保存処理
    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
    def save_regular_tick_data(self):
        """
        通常データの保存処理（当日日付のついた定型ファイル名）
        :return:
        """
        # リアルタイムのタイマー終了後に呼び出される通常保存ファイル名
        name_excel = os.path.join(
            self.res.dir_excel,
            f"tick_{self.dict_ts["date_str"]}.xlsx"
        )
        # Trader インスタンスからティックデータのデータフレームを辞書で取得
        dict_df = self.get_current_tick_data()

        # 念のため、空のデータでないか確認して空でなければ保存
        r = 0
        for code in dict_df.keys():
            df = dict_df[code]
            r += len(df)
        if r == 0:
            # すべてのデータフレームの行数が 0 の場合は保存しない。
            self.logger.info(f"{__name__}: cancel saving {name_excel}, since no data exists.")
            return False
        else:
            # ティックデータの保存処理
            self.save_tick_data(name_excel, dict_df)
            return True

    def save_tick_data(self, name_excel: str, dict_df: dict):
        """
        指定されたファイル名で辞書に格納されたデータフレームExcelシートにしてブックで保存
        :param name_excel:
        :param dict_df:
        :return:
        """
        try:
            save_dataframe_to_excel(name_excel, dict_df)
            self.logger.info(f"{__name__} tick date is saved to {name_excel}.")
        except ValueError as e:
            self.logger.error(f"{__name__} error occurred!: {e}")

    ###########################################################################
    #
    # デバッグ（レビュー）用メソッド
    #
    ###########################################################################
    def on_create_thread_review(self, excel_path: str):
        """
        レビュー用ティックデータ取得スレッドの生成
        :param excel_path:
        :return:
        """
        # ザラ場の開始時間などのタイムスタンプ取得（Excelの日付）
        self.dict_ts = get_intraday_timestamp(excel_path)
        # ---------------------------------------------------------------------
        # 00. デバッグ/レビュー用データ取得インスタンスの生成
        self.worker = ExcelReviewWorker(excel_path)
        self.worker.moveToThread(self.thread)
        # ---------------------------------------------------------------------
        # 01. データ読み込み済みの通知（レビュー用のみ）
        self.worker.notifyDataReady.connect(self.set_data_ready_status)
        # =====================================================================
        # 02. スレッドが開始されたら、ワーカースレッド内で初期化処理を実行するシグナルを発行
        self.thread.started.connect(self.requestWorkerInit.emit)
        # ---------------------------------------------------------------------
        # 03. 初期化処理は指定された Excel ファイルの読み込み
        self.requestWorkerInit.connect(self.worker.initWorker)
        # ---------------------------------------------------------------------
        # 04. 売買ポジション処理用のメソッドへキューイング
        self.requestPositionOpen.connect(self.worker.posman.openPosition)
        self.requestPositionClose.connect(self.worker.posman.closePosition)
        # ---------------------------------------------------------------------
        # 05. 取引結果を取得するメソッドへキューイング
        self.requestTransactionResult.connect(self.worker.getTransactionResult)
        # ---------------------------------------------------------------------
        # 06. 現在株価を取得するメソッドへキューイング。
        self.requestCurrentPrice.connect(self.worker.readCurrentPrice)
        # ---------------------------------------------------------------------
        # 07. スレッドを終了する下記のメソッドへキューイング（リアルタイムでは xlwings 関連）。
        self.requestStopProcess.connect(self.worker.stopProcess)
        # =====================================================================
        # 08. 初期化後の銘柄情報を通知
        self.worker.notifyTickerN.connect(self.on_create_trader)
        # ---------------------------------------------------------------------
        # 09. タイマーで現在時刻と株価を通知
        self.worker.notifyCurrentPrice.connect(self.on_update_data)
        # ---------------------------------------------------------------------
        # 10. 取引結果を通知
        self.worker.notifyTransactionResult.connect(self.on_transaction_result)
        # ---------------------------------------------------------------------
        # 11. スレッド終了関連
        self.worker.threadFinished.connect(self.on_thread_finished)
        # =====================================================================
        # 12. スレッドを開始
        self.thread.start()

    def on_request_data_review(self):
        """
        タイマー処理（デバッグ/レビュー用）
        """
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 現在価格の取得要求をワーカースレッドに通知
        self.requestCurrentPrice.emit(self.ts_system)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # システム時間のインクリメント（１秒）
        self.ts_system += 1

        # 取引時間を過ぎたかをチェック
        if self.dict_ts["end_2h"] < self.ts_system <= self.dict_ts["ca"]:
            if not self.finished_trading:
                # ポジションがあればクローズする
                self.force_closing_position()
                # このフラグにより、何回もポジションがあるかどうかの確認を繰り返さない。
                self.finished_trading = True
        elif self.dict_ts["end"] < self.ts_system:
            self.timer.stop()
            self.logger.info(f"{__name__}: timer stopped!")
            # 取引結果を取得
            self.requestTransactionResult.emit()

        # ツールバーの時刻を更新（現在時刻を表示するだけ）
        self.toolbar.updateTime(self.ts_system)

    def on_review_play(self):
        """
        読み込んだデータ・レビュー開始（デバッグ/レビュー用）
        :return:
        """
        if self.flag_data_ready:
            self.ts_system = self.dict_ts["start"]
            # タイマー開始
            self.timer.start()
            self.logger.info(f"{__name__}: timer started!")

    def on_review_stop(self):
        """
        読み込んだデータ・レビュー停止（デバッグ/レビュー用）
        :return:
        """
        if self.timer.isActive():
            self.timer.stop()
            self.logger.info(f"{__name__}: timer stopped!")
            # 取引結果を取得
            self.requestTransactionResult.emit()

    def set_data_ready_status(self, state: bool):
        self.flag_data_ready = state
        self.logger.info(
            f"{__name__}: now, data ready flag becomes {state}!"
        )
