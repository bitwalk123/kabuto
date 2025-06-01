import logging
import os
import sys

from PySide6.QtCore import QThread, QTimer, Signal
from PySide6.QtGui import QIcon, QCloseEvent
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog

from funcs.ios import save_dataframe_to_excel
from funcs.logs import setup_logging
from funcs.uis import clear_boxlayout
from modules.acquisitor import AquireWorker
from modules.trader_pyqtgraph import Trader
from modules.reviewer import ReviewWorker
from structs.res import AppRes
from widgets.containers import Widget
from widgets.layouts import VBoxLayout
from widgets.statusbar import StatusBar
from widgets.toolbar import ToolBar

if sys.platform == "win32":
    from pywintypes import com_error  # Windows 固有のライブラリ

    debug = False
else:
    debug = True


class Kabuto(QMainWindow):
    __app_name__ = "Kabuto"
    __version__ = "0.1.0"

    request_acquire_init = Signal()
    request_review_init = Signal()

    def __init__(self, options: list = None):
        super().__init__()
        global debug
        self.res = res = AppRes()

        # コンソールから起動した際のオプション・チェック
        if len(options) > 0:
            for option in options:
                if option == "debug":
                    debug = True

        # モジュール固有のロガーを取得
        self.logger = logging.getLogger(__name__)

        if debug:
            self.logger.info(f"{__name__} executed as DEBUG mode!")

            # ウィンドウ・タイトル（デバッグモード）文字列
            title_window = f"{self.__app_name__} - {self.__version__} [debug mode]"

            # タイマー間隔（ミリ秒）（デバッグ時）
            self.timer_interval = 100

            # タイマー開始用フラグ（データ読込済か？）
            self.data_ready = False
            # タイマー用カウンター（レビュー用）
            self.ts_current = 0
            self.ts_start = 0  # タイマー開始時
            self.ts_end = 0  # タイマー終了時
        else:
            self.logger.info(f"{__name__} executed as NORMAL mode!")

            # ウィンドウ・タイトル文字列
            title_window = f"{self.__app_name__} - {self.__version__}"

            # タイマー間隔（ミリ秒）
            self.timer_interval = 1000

        # デバッグ・モードを保持
        res.debug = debug

        # ザラ場用インスタンス（スレッド）
        self.acquire_thread: QThread | None = None
        self.acquire: AquireWorker | None = None

        # Excel レビュー用インスタンス（スレッド）
        self.review_thread: QThread | None = None
        self.review: ReviewWorker | None = None

        # ticker インスタンスを保持する辞書
        self.dict_trader = dict()

        # ウィンドウアイコンとタイトルを設定
        icon = QIcon(os.path.join(res.dir_image, "kabuto.png"))
        self.setWindowIcon(icon)
        self.setWindowTitle(title_window)

        # ツールバー
        toolbar = ToolBar(res)
        toolbar.excelSelected.connect(self.on_create_review_thread)
        toolbar.playClicked.connect(self.on_play)
        toolbar.saveClicked.connect(self.on_save)
        toolbar.stopClicked.connect(self.on_stop)
        self.addToolBar(toolbar)

        # メインウィジェット
        base = Widget()
        self.setCentralWidget(base)
        self.layout = layout = VBoxLayout()
        base.setLayout(layout)

        # ステータス・バー
        self.statusbar = statusbar = StatusBar(res)
        self.setStatusBar(statusbar)

        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # タイマー
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        self.timer = timer = QTimer()
        timer.setInterval(self.timer_interval)
        if debug:
            timer.timeout.connect(self.on_request_data_review)
        else:
            self.on_create_acquire_thread("targets.xlsx")

    def closeEvent(self, event: QCloseEvent):
        if self.timer.isActive():
            self.timer.stop()

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

    def on_create_acquire_thread(self, excel_path: str):
        """
        RSS が書き込んだ銘柄、株価情報を読み取るワーカースレッドを作成

        このスレッドは QThread の　run メソッドを継承していないので、
        明示的にワーカースレッドを終了する処理をしない限り残っていてイベント待機状態になっている。

        :param excel_path:
        :return:
        """
        # Excelを読み込むスレッド処理
        self.acquire_thread = acquire_thread = QThread()
        self.acquire = acquire = AquireWorker(excel_path)
        acquire.moveToThread(acquire_thread)

        # QThread が開始されたら、ワーカースレッド内で初期化処理を開始するシグナルを発行
        acquire_thread.started.connect(self.request_acquire_init.emit)
        # 初期化処理は指定された Excel ファイルを読み込むこと
        self.request_acquire_init.connect(acquire.loadExcel)

        # シグナルとスロットの接続
        acquire.notifyTickerN.connect(self.on_create_trader_acquire)
        # acquire.notifyNewData.connect(self.on_update_data)
        acquire.threadFinished.connect(self.on_thread_finished)
        acquire.threadFinished.connect(acquire_thread.quit)  # スレッド終了時
        acquire_thread.finished.connect(acquire_thread.deleteLater)  # スレッドオブジェクトの削除

        # スレッドを開始
        self.acquire_thread.start()

    def on_create_review_thread(self, excel_path: str):
        """
        保存したティックデータをレビューするためのワーカースレッドを作成

        このスレッドは QThread の　run メソッドを継承していないので、
        明示的にワーカースレッドを終了する処理をしない限り残っていてイベント待機状態になっている。

        :param excel_path:
        :return:
        """
        # Excelを読み込むスレッド処理
        self.review_thread = review_thread = QThread()
        self.review = review = ReviewWorker(excel_path)
        review.moveToThread(review_thread)

        # QThread が開始されたら、ワーカースレッド内で初期化処理を開始するシグナルを発行
        review_thread.started.connect(self.request_review_init.emit)
        # 初期化処理は指定された Excel ファイルを読み込むこと
        self.request_review_init.connect(review.loadExcel)

        # シグナルとスロットの接続
        review.notifyTickerN.connect(self.on_create_trader_review)
        review.notifyNewData.connect(self.on_update_data)
        review.threadFinished.connect(self.on_thread_finished)
        review.threadFinished.connect(review_thread.quit)  # スレッド終了時
        review_thread.finished.connect(review_thread.deleteLater)  # スレッドオブジェクトの削除

        # スレッドを開始
        self.review_thread.start()

    def on_create_trader_acquire(self, list_ticker: list, dict_name: dict, dict_lastclose: dict):
        print(list_ticker)
        print(dict_name)
        print(dict_lastclose)

    def on_create_trader_review(self, list_ticker: list, dict_times: dict):
        # 配置済みの Trader インスタンスを消去
        clear_boxlayout(self.layout)

        # Trader 辞書のクリア
        self.dict_trader = dict()

        # Trader の配置
        for i, ticker in enumerate(list_ticker):
            trader = Trader(self.res)

            # Trader 辞書に保持
            self.dict_trader[ticker] = trader

            # ticker をタイトルに
            trader.setTitle(ticker)
            # チャートの時間範囲を設定
            trader.setTimeRange(*dict_times[ticker])

            self.layout.addWidget(trader)

            # ループ用処理
            if i == 0:
                self.ts_start, self.ts_end = dict_times[ticker]

        # データ読込済フラグを立てる
        self.data_ready = True

    def on_play(self):
        if self.data_ready:
            self.ts_current = self.ts_start
            self.timer.start()

    def on_save(self) -> bool:
        dict_df = dict()
        for ticker in self.dict_trader.keys():
            trader = self.dict_trader[ticker]
            dict_df[ticker] = trader.getTimePrice()

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

    def on_stop(self):
        if self.timer.isActive():
            self.timer.stop()

    def on_thread_finished(self, result: bool):
        if result:
            print("スレッドが正常終了しました。")
        else:
            print("スレッドが異常終了しました。")

        if self.timer.isActive():
            self.timer.stop()

    def on_request_data_review(self):
        self.review.requestNewData(self.ts_current)
        self.ts_current += 1
        if self.ts_end < self.ts_current:
            self.timer.stop()

    def on_update_data(self, dict_data):
        for ticker in dict_data.keys():
            x, y = dict_data[ticker]
            if y > 0:
                trader = self.dict_trader[ticker]
                trader.setTimePrice(x, y)

    def save_tick_data(self, name_excel: str, dict_df: dict):
        try:
            save_dataframe_to_excel(name_excel, dict_df)
            self.logger.info(f"{__name__} データが {name_excel} に保存されました。")
        except ValueError as e:
            self.logger.error(f"{__name__} error occured!: {e}")


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
