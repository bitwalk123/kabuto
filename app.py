import datetime
import logging
import os
import re
import sys
import time

from PySide6.QtCore import QThread, QTimer, Signal
from PySide6.QtGui import QIcon, QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
)

from funcs.ios import save_dataframe_to_excel
from funcs.logs import setup_logging
from funcs.uis import clear_boxlayout
from modules.acquisitor import AquireWorker
from modules.trader_pyqtgraph import Trader
from modules.reviewer import ReviewWorker
from structs.res import AppRes
from widgets.containers import Widget
from widgets.dialog import DlgAboutThis
from widgets.layouts import VBoxLayout
from widgets.statusbar import StatusBar
from widgets.toolbar import ToolBar

if sys.platform == "win32":
    debug = False
else:
    # Windows でないプラットフォーム上ではデバッグモードになる
    debug = True


class Kabuto(QMainWindow):
    __app_name__ = "Kabuto"
    __version__ = "0.4.0"
    __author__ = "Fuhito Suguri"
    __license__ = "MIT"

    # ランタイム用
    requestAcquireInit = Signal()
    requestCurrentPrice = Signal()
    requestStopProcess = Signal()

    # デバッグ用
    requestReviewInit = Signal()
    requestCurrentPriceReview = Signal(float)

    def __init__(self, options: list = None):
        super().__init__()
        global debug
        self.res = res = AppRes()

        # コンソールから起動した際のオプション・チェック
        if len(options) > 0:
            for option in options:
                if option == "debug":
                    # 主に Windows 上でデバッグモードを使用する場合
                    debug = True

        # モジュール固有のロガーを取得
        self.logger = logging.getLogger(__name__)

        #######################################################################
        # ランタイム／デバッグ・モードの設定
        # Windows 以外は常にデバッグ・モード
        # Windows 上でも debug を引数にしてアプリを起動すればデバッグ・モード
        if debug:
            # デバッグ・モードで起動
            self.logger.info(f"{__name__} executed as DEBUG mode!")

            # ウィンドウ・タイトル（デバッグモード）文字列
            title_window = f"{self.__app_name__} - {self.__version__} [debug mode]"

            # タイマー間隔（ミリ秒）（デバッグ時）
            self.timer_interval = 100

            # タイマー開始用フラグ（データ読込済か？）
            self.data_ready = False
        else:
            # ランタイム・モードで起動
            self.logger.info(f"{__name__} executed as NORMAL mode!")

            # ウィンドウ・タイトル文字列
            title_window = f"{self.__app_name__} - {self.__version__}"

            # タイマー間隔（ミリ秒）
            self.timer_interval = 1000

        # ---------------------------------------------------------------------
        # デバッグ・モードを保持
        res.debug = debug
        #
        #######################################################################

        # システム時刻（タイムスタンプ）
        self.ts_system = 0

        # ザラ場中の時刻情報の初期化
        self.ts_start = 0
        self.ts_end_1h = 0
        self.ts_start_2h = 0
        self.ts_ca = 0
        self.ts_end = 0
        self.date_str = 0

        # ザラ場用インスタンス（ランタイム・スレッド）
        self.acquire_thread: QThread | None = None
        self.acquire: AquireWorker | None = None

        # Excel レビュー用インスタンス（デバッグ・スレッド）
        self.review_thread: QThread | None = None
        self.review: ReviewWorker | None = None

        # ticker インスタンスを保持する辞書
        self.dict_trader = dict()

        # ---------------------------------------------------------------------
        #  UI
        # ---------------------------------------------------------------------

        # ウィンドウアイコンとタイトルを設定
        icon = QIcon(os.path.join(res.dir_image, "kabuto.png"))
        self.setWindowIcon(icon)
        self.setWindowTitle(title_window)

        # ツールバー
        self.toolbar = toolbar = ToolBar(res)
        toolbar.aboutClicked.connect(self.on_about)
        toolbar.excelSelected.connect(self.on_create_review_thread)
        toolbar.playClicked.connect(self.on_review_play)
        toolbar.saveClicked.connect(self.on_save_data)
        toolbar.stopClicked.connect(self.on_review_stop)
        toolbar.timerIntervalChanged.connect(self.on_timer_interval_changed)
        self.addToolBar(toolbar)

        # メインウィジェット
        base = Widget()
        self.setCentralWidget(base)
        self.layout = layout = VBoxLayout()
        base.setLayout(layout)

        # ステータス・バー
        self.statusbar = statusbar = StatusBar(res)
        self.setStatusBar(statusbar)

        # タイマー
        self.timer = timer = QTimer()
        timer.setInterval(self.timer_interval)

        if debug:
            # デバッグモードではファイルを読み込んでからスレッドを起動
            timer.timeout.connect(self.on_request_data_review)
        else:
            # リアルタイムモードでは、直ちにスレッドを起動
            timer.timeout.connect(self.on_request_data)
            self.on_create_acquire_thread("targets.xlsx")

    def closeEvent(self, event: QCloseEvent):
        """
        アプリ終了イベント
        :param event:
        :return:
        """
        if self.timer.isActive():
            self.timer.stop()
            self.logger.info("タイマーを停止しました。")

        if self.acquire_thread is not None:
            try:
                if self.acquire_thread.isRunning():
                    self.requestStopProcess.emit()
                    self.acquire_thread.quit()
                    self.acquire_thread.deleteLater()
                    self.logger.info(f"acquire スレッドを削除しました。")
            except RuntimeError as e:
                self.logger.info(f"終了時: {e}")

        if self.review_thread is not None:
            try:
                if self.review_thread.isRunning():
                    self.review_thread.quit()
                    self.review_thread.deleteLater()
                    self.logger.info(f"reviewer スレッドを削除しました。")
            except RuntimeError as e:
                self.logger.info(f"終了時: {e}")

        self.logger.info(f"{__name__} stopped and closed.")
        event.accept()

    def create_trader(self, list_ticker, dict_name, dict_lastclose):
        """
        銘柄数分の Trader インスタンスの生成
        :param list_ticker:
        :param dict_name:
        :param dict_lastclose:
        :return:
        """
        # 配置済みの Trader インスタンスを消去
        clear_boxlayout(self.layout)

        # Trader 辞書のクリア
        self.dict_trader = dict()

        # 銘柄数分の Trader インスタンスの生成
        for ticker in list_ticker:
            # Trader インスタンスの生成
            trader = Trader(self.res, ticker)
            trader.dock.clickedBuy.connect(self.on_buy)
            trader.dock.clickedRepay.connect(self.on_repay)
            trader.dock.clickedSell.connect(self.on_sell)

            # Trader 辞書に保持
            self.dict_trader[ticker] = trader

            # 「銘柄名　(ticker)」をタイトルにして設定し直し
            trader.setTitle(f"{dict_name[ticker]} ({ticker})")

            # 当日ザラ場時間
            trader.setTimeRange(self.ts_start, self.ts_end)

            # 前日終値
            if dict_lastclose[ticker] > 0:
                trader.addLastCloseLine(dict_lastclose[ticker])

            # 配置
            self.layout.addWidget(trader)

    def get_current_tick_data(self) -> dict:
        """
        チャートが保持しているティックデータをデータフレームで取得
        :return:
        """
        dict_df = dict()
        for ticker in self.dict_trader.keys():
            trader = self.dict_trader[ticker]
            dict_df[ticker] = trader.getTimePrice()
        return dict_df

    def on_about(self):
        """
        このアプリについて（ダイアログ表示）
        :return:
        """
        dlg = DlgAboutThis(
            self.res,
            self.__app_name__,
            self.__version__,
            self.__author__,
            self.__license__
        )
        if dlg.exec():
            print('OK ボタンがクリックされました。')

    def on_create_acquire_thread(self, excel_path: str):
        """
        RSS が書き込んだ銘柄、株価情報を読み取るワーカースレッドを作成

        このスレッドは QThread の　run メソッドを継承していないので、
        明示的にワーカースレッドを終了する処理をしない限り残っていてイベント待機状態になっている。

        :param excel_path:
        :return:
        """
        # _____________________________________________________________________
        # 本日の日時情報より、チャートのx軸の始点と終点を算出
        dt = datetime.datetime.now()
        year = dt.year
        month = dt.month
        day = dt.day
        # ザラ場日時時間情報
        self.set_intraday_time(year, month, day)

        # _____________________________________________________________________
        # Excelを読み込むスレッド処理
        self.acquire_thread = acquire_thread = QThread()
        self.acquire = acquire = AquireWorker(excel_path)
        acquire.moveToThread(acquire_thread)

        # QThread が開始されたら、ワーカースレッド内で初期化処理を開始するシグナルを発行
        acquire_thread.started.connect(self.requestAcquireInit.emit)
        # _____________________________________________________________________
        # メイン・スレッド側のシグナルとワーカー・スレッド側のスロット（メソッド）の接続
        # 初期化処理は指定された Excel ファイルを読み込むこと
        # xlwings インスタンスを生成、Excel の銘柄情報を読込むメソッドへキューイング。
        self.requestAcquireInit.connect(acquire.loadExcel)

        # 現在株価を取得するメソッドへキューイング。
        self.requestCurrentPrice.connect(acquire.readCurrentPrice)

        # xlwings インスタンスを破棄、スレッドを終了する下記のメソッドへキューイング。
        self.requestStopProcess.connect(acquire.stopProcess)

        # _____________________________________________________________________
        # ワーカー・スレッド側のシグナルとスロットの接続
        acquire.notifyTickerN.connect(self.on_create_trader)
        acquire.notifyCurrentPrice.connect(self.on_update_data)
        acquire.threadFinished.connect(self.on_thread_finished)
        acquire.threadFinished.connect(acquire_thread.quit)  # スレッド終了時
        acquire_thread.finished.connect(acquire_thread.deleteLater)  # スレッドオブジェクトの削除

        # _____________________________________________________________________
        # スレッドを開始
        self.acquire_thread.start()

    def on_create_trader(self, list_ticker: list, dict_name: dict, dict_lastclose: dict):
        """
        Trader インスタンスの生成（リアルタイム）
        :param list_ticker:
        :param dict_name:
        :param dict_lastclose:
        :return:
        """
        # 銘柄数分の Trader インスタンスの生成
        self.create_trader(list_ticker, dict_name, dict_lastclose)

        # リアルタイムの場合はここでタイマーを開始
        self.timer.start()
        self.logger.info("タイマーを開始しました。")

    def on_request_data(self):
        """
        タイマー処理（リアルタイム）
        """
        self.ts_system = time.time()
        if self.ts_start <= self.ts_system <= self.ts_end_1h:
            self.requestCurrentPrice.emit()
        elif self.ts_start_2h <= self.ts_system <= self.ts_ca:
            self.requestCurrentPrice.emit()
        elif self.ts_ca < self.ts_system:
            self.timer.stop()
            self.logger.info("タイマーを停止しました。")
            self.save_regular_tick_data()
        else:
            pass

        # ツールバーの時刻を更新
        self.toolbar.updateTime(self.ts_system)

    def on_save_data(self) -> bool:
        """
        チャートが保持しているデータをファイル名を指定して保存
        :return:
        """
        dict_df = self.get_current_tick_data()

        name_excel, _ = QFileDialog.getSaveFileName(
            self,
            "ティック・データを保存",
            "Unknown.xlsx",
            "Excel Files (*.xlsx);;All Files (*)",
            "Excel Files (*.xlsx)"
        )
        if name_excel == "":
            return False
        else:
            print(name_excel)
            self.save_tick_data(name_excel, dict_df)
            return True

    def on_thread_finished(self, result: bool):
        """
        スレッド終了時のログ
        :param result:
        :return:
        """
        if result:
            self.logger.info("スレッドが正常終了しました。")
        else:
            self.logger.error("スレッドが異常終了しました。")

        if self.timer.isActive():
            self.timer.stop()
            self.logger.info("タイマーを停止しました。")

    def on_update_data(self, dict_data):
        """
        ティックデータの更新
        :param dict_data:
        :return:
        """
        for ticker in dict_data.keys():
            x, y = dict_data[ticker]
            trader = self.dict_trader[ticker]
            trader.setTimePrice(x, y)

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
            f"tick_{self.date_str}.xlsx"
        )
        # Trader インスタンスからティックデータのデータフレームを辞書で取得
        dict_df = self.get_current_tick_data()

        # 念のため、空のデータでないか確認して空でなければ保存
        r = 0
        for ticker in dict_df.keys():
            df = dict_df[ticker]
            r += len(df)
        if r == 0:
            # すべてのデータフレームの行数が 0 の場合は保存しない。
            self.logger.info(f"{__name__} データ無いため {name_excel} への保存はキャンセルされました。")
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
            self.logger.info(f"{__name__} データが {name_excel} に保存されました。")
        except ValueError as e:
            self.logger.error(f"{__name__} error occured!: {e}")

    def set_intraday_time(self, year, month, day):
        """
        ザラ場日時時間情報設定

        :param year:
        :param month:
        :param day:
        :return:
        """
        dt_start = datetime.datetime(year, month, day, hour=9, minute=0)
        dt_end_1h = datetime.datetime(year, month, day, hour=11, minute=30)
        dt_start_2h = datetime.datetime(year, month, day, hour=12, minute=30)
        dt_ca = datetime.datetime(year, month, day, hour=15, minute=25)
        dt_end = datetime.datetime(year, month, day, hour=15, minute=30)
        # タイムスタンプに変換してインスタンス変数で保持
        self.ts_start = dt_start.timestamp()
        self.ts_end_1h = dt_end_1h.timestamp()
        self.ts_start_2h = dt_start_2h.timestamp()
        self.ts_ca = dt_ca.timestamp()
        self.ts_end = dt_end.timestamp()
        # 日付文字列
        self.date_str = f"{year:04}{month:02}{day:02}"

    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
    # 取引ボタンがクリックされた時の処理
    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
    def on_sell(self, ticker, price):
        print(f"clicked SELL button at {ticker} {price}")

    def on_buy(self, ticker, price):
        print(f"clicked BUY button at {ticker} {price}")

    def on_repay(self, ticker, price):
        print(f"clicked REPAY button at {ticker} {price}")

    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
    # デバッグ用メソッド
    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
    def on_create_review_thread(self, excel_path: str):
        """
        保存されたティックデータをレビューするためのワーカースレッドを作成（デバッグ用）

        このスレッドは QThread の　run メソッドを継承していないので、
        明示的にワーカースレッドを終了する処理をしない限り残っていてイベント待機状態になっている。

        :param excel_path:
        :return:
        """
        # _____________________________________________________________________
        # Excel のファイル名より、チャートのx軸の始点と終点を算出
        pattern = re.compile(r".*tick_([0-9]{4})([0-9]{2})([0-9]{2})\.xlsx")
        m = pattern.match(excel_path)
        if m:
            year = int(m.group(1))
            month = int(m.group(2))
            day = int(m.group(3))
        else:
            year = 1970
            month = 1
            day = 1
        # ザラ場日時時間情報
        self.set_intraday_time(year, month, day)

        # Excelを読み込むスレッド処理
        self.review_thread = review_thread = QThread()
        self.review = review = ReviewWorker(excel_path)
        review.moveToThread(review_thread)

        # QThread が開始されたら、ワーカースレッド内で初期化処理を開始するシグナルを発行
        review_thread.started.connect(self.requestReviewInit.emit)
        # 初期化処理は指定された Excel ファイルを読み込むこと
        self.requestReviewInit.connect(review.loadExcel)

        # 現在株価を取得するメソッドへキューイング。
        self.requestCurrentPriceReview.connect(review.readCurrentPrice)

        # シグナルとスロットの接続
        review.notifyTickerN.connect(self.on_create_trader_review)
        review.notifyCurrentPrice.connect(self.on_update_data_review)
        review.threadFinished.connect(self.on_thread_finished)
        review.threadFinished.connect(review_thread.quit)  # スレッド終了時
        review_thread.finished.connect(review_thread.deleteLater)  # スレッドオブジェクトの削除

        # スレッドを開始
        self.review_thread.start()

    def on_create_trader_review(self, list_ticker: list, dict_name: dict, dict_lastclose: dict):
        """
        Trader インスタンスの生成（デバッグ用）
        :param list_ticker:
        :param dict_times:
        :return:
        """
        # 銘柄数分の Trader インスタンスの生成
        self.create_trader(list_ticker, dict_name, dict_lastclose)

        # デバッグの場合はスタート・ボタンがクリックされるまでは待機
        self.data_ready = True
        self.logger.info("レビューの準備完了です。")

    def on_request_data_review(self):
        """
        タイマー処理（デバッグ）
        """
        self.requestCurrentPriceReview.emit(self.ts_system)
        self.ts_system += 1
        if self.ts_end < self.ts_system:
            self.timer.stop()
            self.logger.info("タイマーを停止しました。")

        # ツールバーの時刻を更新（現在時刻を表示するだけ）
        self.toolbar.updateTime(self.ts_system)

    def on_review_play(self):
        """
        読み込んだデータ・レビュー開始（デバッグ用）
        :return:
        """
        if self.data_ready:
            self.ts_system = self.ts_start
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
            self.logger.info("タイマーを停止しました。")

    def on_timer_interval_changed(self, interval: int):
        """
        レビュー用タイマー間隔の変更（デバッグ用）
        :param interval:
        :return:
        """
        self.logger.info(f"タイマー間隔が {interval} ミリ秒に設定されました。")
        self.timer.setInterval(interval)

    def on_update_data_review(self, dict_data):
        """
        ティックデータの更新（デバッグ用）
        :param dict_data:
        :return:
        """
        for ticker in dict_data.keys():
            x, y = dict_data[ticker]
            if y > 0:
                # 該当するデータが無い場合は 0 が帰ってくるため
                trader = self.dict_trader[ticker]
                trader.setTimePrice(x, y)


def main():
    app = QApplication(sys.argv)
    options = sys.argv[1:]
    win = Kabuto(options)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    # ロギング設定を適用（ルートロガーを設定）
    main_logger = setup_logging()
    # main_logger.info("Application starting up and logging initialized.")
    main()
