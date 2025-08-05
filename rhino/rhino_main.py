import logging
import os
import time

import pandas as pd
from PySide6.QtCore import QTimer
from PySide6.QtGui import QIcon, QCloseEvent
from PySide6.QtWidgets import QMainWindow

from funcs.ios import save_dataframe_to_excel
from funcs.uis import clear_boxlayout
from modules.trans import WinTransaction
from rhino.rhino_acquire import RhinoAcquire
from rhino.rhino_dialog import DlgAboutThis
from rhino.rhino_dock import DockRhinoTrader
from rhino.rhino_funcs import get_intraday_timestamp
from rhino.rhino_psar import PSARObject
from rhino.rhino_review import RhinoReview
from rhino.rhino_statusbar import RhinoStatusBar
from rhino.rhino_ticker import Ticker
from rhino.rhino_toolbar import RhinoToolBar
from rhino.rhino_trader import RhinoTrader
from structs.app_enum import PositionType
from structs.res import AppRes
from widgets.containers import Widget
from widgets.layouts import VBoxLayout


class Rhino(QMainWindow):
    __app_name__ = "Rhino"
    __version__ = "0.9.1"
    __author__ = "Fuhito Suguri"
    __license__ = "MIT"

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
        else:
            # NORMAL モード
            self.logger.info(f"{__name__}: executed as NORMAL mode!")
            self.timer_interval = 1000  # タイマー間隔（ミリ秒）
        #
        #######################################################################

        # 株価取得スレッド用インスタンス
        self.acquire: RhinoAcquire | None = None
        self.review: RhinoReview | None = None

        # システム時刻（タイムスタンプ）
        self.ts_system = 0

        # ザラ場の開始時間などのタイムスタンプ取得（本日分）
        self.dict_ts = get_intraday_timestamp()

        # 取引が終了したかどうかのフラグ
        self.finished_trading = False

        # trader インスタンスを保持する辞書
        self.dict_trader = dict()

        # ThreadTicker 用インスタンス
        self.ticker: Ticker | None = None
        # ThreadTicker インスタンスを保持する辞書
        self.dict_ticker = dict()

        # 取引履歴
        self.df_transaction = None
        self.win_transaction: WinTransaction | None = None

        # ---------------------------------------------------------------------
        #  UI
        # ---------------------------------------------------------------------
        # ウィンドウアイコンとタイトルを設定
        icon = QIcon(os.path.join(res.dir_image, "rhino.png"))
        self.setWindowIcon(icon)
        title_win = f"{self.__app_name__} - {self.__version__}"
        if debug:
            title_win = f"{title_win} [debug mode]"
        self.setWindowTitle(title_win)

        # ツールバー
        self.toolbar = toolbar = RhinoToolBar(res)
        toolbar.clickedAbout.connect(self.on_about)
        toolbar.clickedPlay.connect(self.on_review_play)
        toolbar.clickedStop.connect(self.on_review_stop)
        toolbar.selectedExcelFile.connect(self.on_create_review_thread)
        toolbar.clickedTransaction.connect(self.on_show_transaction)
        self.addToolBar(toolbar)

        # ステータスバー
        self.statusbar = statusbar = RhinoStatusBar(res)
        self.setStatusBar(statusbar)

        # メイン・ウィジェット
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
            self.on_create_acquire_thread(excel_path)

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
        # self.acquire スレッドの削除
        # ---------------------------------------------------------------------
        if self.acquire is not None:
            try:
                if self.acquire.isRunning():
                    self.acquire.requestStopProcess.emit()
                    time.sleep(1)
                    self.acquire.quit()
                    self.acquire.deleteLater()
                    self.logger.info(f"{__name__}: deleted acquire thread.")
            except RuntimeError as e:
                self.logger.info(f"{__name__}: error at termination: {e}")

        # ---------------------------------------------------------------------
        # self.review スレッドの削除
        # ---------------------------------------------------------------------
        if self.review is not None:
            try:
                if self.review.isRunning():
                    self.review.quit()
                    self.review.deleteLater()
                    self.logger.info(f"{__name__}: deleted review thread.")
            except RuntimeError as e:
                self.logger.info(f"{__name__}: error at termination: {e}")

        # ---------------------------------------------------------------------
        # Thread Ticker の削除
        # ---------------------------------------------------------------------
        for code, thread in self.dict_ticker.items():
            if thread.isRunning():
                self.logger.info(f"{__name__}: stopping ThreadTicker for {code}...")
                thread.quit()  # スレッドのイベントループに終了を指示
                thread.wait()  # スレッドが完全に終了するまで待機
                self.logger.info(f"{__name__}: ThreadTicker for {code} safely terminated.")

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
        # Ticker インスタンスをクリア
        self.dict_ticker = dict()

        # 銘柄数分の Trader および Ticker インスタンスの生成
        for code in list_code:
            # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
            # Trader インスタンスの生成
            # 主にチャート表示用
            # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
            trader = RhinoTrader(self.res, code)
            # Dock の売買ボタンのクリック・シグナルを直接ハンドリング
            if self.res.debug:
                # レビュー用の売買処理
                trader.dock.clickedBuy.connect(self.on_buy_review)
                trader.dock.clickedRepay.connect(self.on_repay_review)
                trader.dock.clickedSell.connect(self.on_sell_review)
            else:
                # リアルタイム用の売買処理
                trader.dock.clickedBuy.connect(self.on_buy)
                trader.dock.clickedRepay.connect(self.on_repay)
                trader.dock.clickedSell.connect(self.on_sell)

            # レビュー/リアルタイム用共通処理
            trader.dock.notifyNewPSARParams.connect(self.notify_new_psar_params)

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

            # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
            # Ticker インスタンスの生成
            # 主に Parabolic SAR の算出用
            # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
            self.ticker = ticker = Ticker(self.res, code)
            ticker.threadReady.connect(self.on_ticker_ready)
            ticker.worker.notifyPSAR.connect(self.on_update_psar)
            ticker.start()
            self.dict_ticker[code] = ticker
            # パラメータ情報をやりとりするために Trader クラスのドックにインスタンスを登録
            trader.dock.setTicker(ticker)

    def force_closing_position(self):
        for code in self.dict_trader.keys():
            trader: RhinoTrader = self.dict_trader[code]
            dock: DockRhinoTrader = trader.dock
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
            self.__license__
        ).exec()

    def on_create_acquire_thread(self, excel_path: str):
        """
        リアルタイム用ティックデータ取得スレッドの生成
        :param excel_path:
        :return:
        """
        # リアルタイム用データ取得インスタンス (self.acquire) の生成
        self.acquire = acquire = RhinoAcquire(excel_path)
        # 初期化後の銘柄情報を通知
        acquire.worker.notifyTickerN.connect(self.on_create_trader)
        # タイマーで現在時刻と株価を通知
        acquire.worker.notifyCurrentPrice.connect(self.on_update_data)
        # 取引結果を通知
        acquire.worker.notifyTransactionResult.connect(self.on_transaction_result)
        # スレッド終了関連
        acquire.worker.threadFinished.connect(self.on_thread_finished)
        # スレッドを開始
        acquire.start()

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

        # ---------------------------------------------------------------------
        # リアルタイムの場合はここでタイマーを開始
        # ---------------------------------------------------------------------
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
            self.acquire.requestCurrentPrice.emit()
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        elif self.dict_ts["start_2h"] <= self.ts_system <= self.dict_ts["end_2h"]:
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # 🧿 現在価格の取得要求をワーカースレッドに通知
            self.acquire.requestCurrentPrice.emit()
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
            self.acquire.requestTransactionResult.emit()
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

    def on_update_data(self, dict_data, dict_profit, dict_total):
        """
        ティックデータ、含み益、損益の更新
        :param dict_data:
        :param dict_profit:
        :param dict_total:
        :return:
        """
        for code in dict_data.keys():
            x, y = dict_data[code]
            trader = self.dict_trader[code]
            trader.dock.setPrice(y)
            # 銘柄単位の含み益と収益を更新
            trader.dock.setProfit(dict_profit[code])
            trader.dock.setTotal(dict_total[code])
            # Parabolic SAR
            ticker: Ticker = self.dict_ticker[code]
            # ここで PSAR を算出する処理が呼び出される
            ticker.notifyNewPrice.emit(x, y)

    def on_update_psar(self, code: str, x: float, ret: PSARObject):
        """
        Parabolic SAR のトレンド点を追加
        :param code:
        :param x:
        :param ret:
        :return:
        """
        trader: RhinoTrader = self.dict_trader[code]
        trader.setPlotData(x, ret)

    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
    # 取引ボタンがクリックされた時の処理（Acquire 用）
    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
    def on_buy(self, code: str, price: float, note: str):
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 買建で建玉取得リクエストのシグナル
        self.acquire.requestPositionOpen.emit(
            code, self.ts_system, price, PositionType.BUY, note
        )
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def on_sell(self, code: str, price: float, note: str):
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 売建で建玉取得リクエストのシグナル
        self.acquire.requestPositionOpen.emit(
            code, self.ts_system, price, PositionType.SELL, note
        )
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def on_repay(self, code: str, price: float, note: str):
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 建玉返済リクエストのシグナル
        self.acquire.requestPositionClose.emit(
            code, self.ts_system, price, note
        )
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def notify_new_psar_params(self, code: str, dict_psar: dict):
        # 銘柄コード別 Parabolic SAR 等の算出用インスタンス
        ticker: Ticker = self.dict_ticker[code]
        # ここで PSAR を算出する処理が呼び出される
        ticker.requestUpdatePSARParams.emit(dict_psar)

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
            f"tick_{self.dict_ts["date_str"]}_rhino.xlsx"
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
    def on_create_review_thread(self, excel_path: str):
        """
        レビュー用ティックデータ取得スレッドの生成
        :param excel_path:
        :return:
        """
        # ザラ場の開始時間などのタイムスタンプ取得（Excelの日付）
        self.dict_ts = get_intraday_timestamp(excel_path)
        # デバッグ/レビュー用データ取得インスタンス (self.review) の生成
        self.review = review = RhinoReview(excel_path)
        # 初期化後の銘柄情報を通知
        review.worker.notifyTickerN.connect(self.on_create_trader_review)
        # タイマーで現在時刻と株価を通知
        review.worker.notifyCurrentPrice.connect(self.on_update_data)
        # 取引結果を通知
        review.worker.notifyTransactionResult.connect(self.on_transaction_result)
        # スレッド終了関連
        review.worker.threadFinished.connect(self.on_thread_finished)
        # スレッドを開始
        review.start()

    def on_create_trader_review(self, list_code: list, dict_name: dict, dict_lastclose: dict):
        """
        Trader インスタンスの生成（デバッグ/レビュー用）
        :param list_code:
        :param dict_name:
        :param dict_lastclose:
        :return:
        """
        # ---------------------------------------------------------------------
        # 銘柄数分の Trader インスタンスの生成
        # ---------------------------------------------------------------------
        self.create_trader(list_code, dict_name, dict_lastclose)

        # ---------------------------------------------------------------------
        # デバッグの場合はスタート・ボタンがクリックされるまでは待機
        # ---------------------------------------------------------------------
        self.logger.info(f"{__name__}: ready to review!")

    def on_request_data_review(self):
        """
        タイマー処理（デバッグ/レビュー用）
        """
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 現在価格の取得要求をワーカースレッドに通知
        self.review.requestCurrentPrice.emit(self.ts_system)
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
            self.review.requestTransactionResult.emit()

        # ツールバーの時刻を更新（現在時刻を表示するだけ）
        self.toolbar.updateTime(self.ts_system)

    def on_review_play(self):
        """
        読み込んだデータ・レビュー開始（デバッグ/レビュー用）
        :return:
        """
        if self.review.isDataReady():
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
            self.review.requestTransactionResult.emit()

    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
    # 取引ボタンがクリックされた時の処理（Review 用）
    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
    def on_buy_review(self, code: str, price: float, note: str):
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 買建で建玉取得リクエストのシグナル
        self.review.requestPositionOpen.emit(
            code, self.ts_system, price, PositionType.BUY, note
        )
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def on_sell_review(self, code: str, price: float, note: str):
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 売建で建玉取得リクエストのシグナル
        self.review.requestPositionOpen.emit(
            code, self.ts_system, price, PositionType.SELL, note
        )
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def on_repay_review(self, code: str, price: float, note: str):
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 建玉返済リクエストのシグナル
        self.review.requestPositionClose.emit(
            code, self.ts_system, price, note
        )
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
