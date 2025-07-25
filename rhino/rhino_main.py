import logging
import os
import sys
import time

import pandas as pd
from PySide6.QtCore import QTimer
from PySide6.QtGui import QIcon, QCloseEvent
from PySide6.QtWidgets import QMainWindow

from funcs.uis import clear_boxlayout
from modules.trans import WinTransaction
from rhino.rhino_dock import DockRhinoTrader
from rhino.rhino_funcs import get_intraday_timestamp
from rhino.rhino_psar import PSARObject
from rhino.rhino_review import RhinoReview
from rhino.rhino_ticker import ThreadTicker
from rhino.rhino_toolbar import RhinoToolBar
from rhino.rhino_trader import RhinoTrader
from structs.posman import PositionType
from structs.res import AppRes
from widgets.containers import Widget
from widgets.layouts import VBoxLayout

if sys.platform == "win32":
    debug = False
else:
    debug = True  # Windows 以外はデバッグ・モード


class Rhino(QMainWindow):
    __app_name__ = "Rhino"
    __version__ = "0.9.0"
    __author__ = "Fuhito Suguri"
    __license__ = "MIT"

    def __init__(self, options: list = None):
        super().__init__()
        global debug  # グローバル変数であることを明示
        self.res = res = AppRes()
        self.logger = logging.getLogger(__name__)  # モジュール固有のロガーを取得

        # コンソールから起動した際のオプションをチェック
        if len(options) > 0:
            for option in options:
                if option == "debug":
                    debug = True  # Windows 上でデバッグ・モードを使用する場合
        # デバッグ・モードを保持
        res.debug = debug

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

        # スレッド用インスタンス
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
        self.thread_ticker: ThreadTicker | None = None
        # ThreadTicker インスタンスを保持する辞書
        self.dict_thread_ticker = dict()

        # 取引履歴
        self.df_transaction: pd.DataFrame | None = None
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
        toolbar.excelSelected.connect(self.on_create_review_thread)
        toolbar.playClicked.connect(self.on_review_play)
        toolbar.stopClicked.connect(self.on_review_stop)
        self.addToolBar(toolbar)

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
            pass
            """
            # リアルタイムモードでは、直ちにスレッドを起動
            timer.timeout.connect(self.on_request_data)
            self.on_create_acquire_thread("targets.xlsx")
            """

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
        """
        if self.acquire is not None:
            try:
                if self.acquire.isRunning():
                    self.requestStopProcess.emit()
                    time.sleep(1)
                    self.acquire.quit()
                    self.acquire.deleteLater()
                    self.logger.info(f"{__name__}: deleted acquire thread.")
            except RuntimeError as e:
                self.logger.info(f"{__name__}: error at termination: {e}")
        """

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
        for ticker, thread in self.dict_thread_ticker.items():
            if thread.isRunning():
                self.logger.info(f"{__name__}: stopping ThreadTicker for {ticker}...")
                thread.quit()  # スレッドのイベントループに終了を指示
                thread.wait()  # スレッドが完全に終了するまで待機
                self.logger.info(f"{__name__}: ThreadTicker for {ticker} safely terminated.")

        # ---------------------------------------------------------------------
        self.logger.info(f"{__name__} stopped and closed.")
        event.accept()

    def create_trader(self, list_ticker, dict_name, dict_lastclose):
        """
        銘柄数分の Trader インスタンスの生成
        （リアルタイム・モード、デバッグ・モード共通）
        :param list_ticker:
        :param dict_name:
        :param dict_lastclose:
        :return:
        """
        # 配置済みの Trader インスタンスを消去
        clear_boxlayout(self.layout)
        # Trader 辞書のクリア
        self.dict_trader = dict()
        # Thread ticker インスタンスをクリア
        self.dict_thread_ticker = dict()

        # 銘柄数分の Trader インスタンスの生成
        for ticker in list_ticker:
            # Trader インスタンスの生成
            trader = RhinoTrader(self.res, ticker)

            # Dock の売買ボタンのクリック・シグナルを直接ハンドリング
            if debug:
                # レビュー用の売買処理
                trader.dock.clickedBuy.connect(self.on_buy_review)
                trader.dock.clickedRepay.connect(self.on_repay_review)
                trader.dock.clickedSell.connect(self.on_sell_review)
            else:
                # リアルタイム用の売買処理
                pass

            # Trader 辞書に保持
            self.dict_trader[ticker] = trader

            # 「銘柄名　(ticker)」をタイトルにして設定し直し
            trader.setChartTitle(f"{dict_name[ticker]} ({ticker})")

            # 当日ザラ場時間
            trader.setTimeAxisRange(self.dict_ts["start"], self.dict_ts["end"])

            # 前日終値
            if dict_lastclose[ticker] > 0:
                trader.setLastCloseLine(dict_lastclose[ticker])

            # 配置
            self.layout.addWidget(trader)

            # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
            # Thread Ticker
            # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
            self.thread_ticker = thread_ticker = ThreadTicker(ticker)
            thread_ticker.threadReady.connect(self.on_thread_ticker_ready)
            thread_ticker.worker.notifyPSAR.connect(self.on_update_psar)
            thread_ticker.start()
            self.dict_thread_ticker[ticker] = thread_ticker

    def force_closing_position(self):
        for ticker in self.dict_trader.keys():
            trader: RhinoTrader = self.dict_trader[ticker]
            dock: DockRhinoTrader = trader.dock
            dock.forceStopAutoPilot()

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

    def on_thread_ticker_ready(self, ticker: str):
        self.logger.info(f"{__name__}: thread for {ticker} is ready.")

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
        for ticker in dict_data.keys():
            x, y = dict_data[ticker]
            trader = self.dict_trader[ticker]
            trader.dock.setPrice(y)
            # 銘柄単位の含み益と収益を更新
            trader.dock.setProfit(dict_profit[ticker])
            trader.dock.setTotal(dict_total[ticker])
            # Parabolic SAR
            thread_ticker: ThreadTicker = self.dict_thread_ticker[ticker]
            # ここで PSAR を算出する処理が呼び出される
            thread_ticker.notifyNewPrice.emit(x, y)

    def on_update_psar(self, ticker: str, x: float, ret: PSARObject):
        """
        Parabolic SAR のトレンド点を追加
        :param ticker:
        :param x:
        :param ret:
        :return:
        """
        trader: RhinoTrader = self.dict_trader[ticker]
        trader.setPlotData(x, ret)

    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
    # 取引ボタンがクリックされた時の処理（レビュー用）
    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
    def on_buy_review(self, ticker: str, price: float, note: str):
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 買建で建玉取得をワーカースレッドに通知
        self.review.requestPositionOpen.emit(
            ticker, self.ts_system, price, PositionType.BUY, note
        )
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def on_sell_review(self, ticker: str, price: float, note: str):
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 売建で建玉取得をワーカースレッドに通知
        self.review.requestPositionOpen.emit(
            ticker, self.ts_system, price, PositionType.SELL, note
        )
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def on_repay_review(self, ticker: str, price: float, note: str):
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 建玉返済をワーカースレッドに通知
        self.review.requestPositionClose.emit(
            ticker, self.ts_system, price, note
        )
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
    # デバッグ用メソッド
    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
    def on_create_review_thread(self, excel_path: str):
        # ザラ場の開始時間などのタイムスタンプ取得（Excelの日付）
        self.dict_ts = get_intraday_timestamp(excel_path)
        # デバッグ用インスタンス (self.review) の作成
        self.review = RhinoReview(excel_path)
        # 初期化後の銘柄情報を通知
        self.review.worker.notifyTickerN.connect(self.on_create_trader_review)
        # タイマーで現在時刻と株価を通知
        self.review.worker.notifyCurrentPrice.connect(self.on_update_data)
        # 取引結果を通知
        self.review.worker.notifyTransactionResult.connect(self.on_transaction_result)
        # スレッド終了関連
        self.review.worker.threadFinished.connect(self.on_thread_finished)

        # スレッドを開始
        self.review.start()

    def on_create_trader_review(self, list_ticker: list, dict_name: dict, dict_lastclose: dict):
        """
        Trader インスタンスの生成（デバッグ用）
        :param list_ticker:
        :param dict_name:
        :param dict_lastclose:
        :return:
        """
        # 銘柄数分の Trader インスタンスの生成
        self.create_trader(list_ticker, dict_name, dict_lastclose)

        # デバッグの場合はスタート・ボタンがクリックされるまでは待機
        # self.data_ready = True
        self.logger.info(f"{__name__}: ready to review!")

    def on_request_data_review(self):
        """
        タイマー処理（デバッグ）
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
            self.logger.info(f"Timer stopped!")
            # 取引結果を取得
            self.review.requestTransactionResult.emit()

        # ツールバーの時刻を更新（現在時刻を表示するだけ）
        self.toolbar.updateTime(self.ts_system)

    def on_review_play(self):
        """
        読み込んだデータ・レビュー開始（デバッグ用）
        :return:
        """
        if self.review.isDataReady():
            self.ts_system = self.dict_ts["start"]
            # タイマー開始
            self.timer.start()
            self.logger.info("タイマーを開始しました。")

    def on_review_stop(self):
        """
        読み込んだデータ・レビュー停止（デバッグ用）
        :return:
        """
        if self.timer.isActive():
            self.timer.stop()
            self.logger.info(f"{__name__}: timer stopped!")
            # 取引結果を取得
            self.review.requestTransactionResult.emit()
