import datetime
import logging
import os
import time

from PySide6.QtCore import QTimer
from PySide6.QtGui import QCloseEvent, QIcon
from PySide6.QtWidgets import QMainWindow

from structs.res import AppRes
from vein.collector import StockCollector
from widgets.toolbars import ToolBarVein


class StockVein(QMainWindow):
    def __init__(self):
        super().__init__()
        # モジュール固有のロガーを取得
        self.logger = logging.getLogger(__name__)

        self.res = res = AppRes()
        self.timer_interval = 1000

        # _____________________________________________________________________
        # 本日の日時情報より、ザラバ中の時刻情報を定義
        dt = datetime.datetime.now()
        year = dt.year
        month = dt.month
        day = dt.day
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

        # ---------------------------------------------------------------------
        #  UI
        # ---------------------------------------------------------------------
        # ウィンドウアイコンとタイトルを設定
        icon = QIcon(os.path.join(res.dir_image, "vein.png"))
        self.setWindowIcon(icon)
        self.setWindowTitle("StockVein")

        # ツールバー
        self.toolbar = toolbar = ToolBarVein(res)
        self.addToolBar(toolbar)

        self.stock_collector = stock_collector = StockCollector(res)
        stock_collector.threadReady.connect(self.on_stock_collector_ready)
        stock_collector.worker.notifyTickerN.connect(self.on_ticker_list)
        stock_collector.worker.saveCompleted.connect(self.on_save_completed)
        stock_collector.start()

        self.timer = timer = QTimer()
        timer.setInterval(self.timer_interval)
        timer.timeout.connect(self.on_request_data)
        timer.start()

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
            self.logger.info("タイマーを停止しました。")

        # ---------------------------------------------------------------------
        # Thread Stock Collector の削除
        # ---------------------------------------------------------------------
        if self.stock_collector.isRunning():
            self.stock_collector.requestStopProcess.emit()
            self.logger.info("Stopping StockCollector...")
            self.stock_collector.quit()  # スレッドのイベントループに終了を指示
            self.stock_collector.wait()  # スレッドが完全に終了するまで待機
            self.logger.info("StockCollector safely terminated.")

        # ---------------------------------------------------------------------
        self.logger.info(f"{__name__} stopped and closed.")
        event.accept()

    def on_request_data(self):
        # システム時刻
        self.ts_system = time.time()
        if self.ts_start <= self.ts_system <= self.ts_end_1h:
            # ----------------------------------------------
            # 🧿 現在価格の取得要求をスレッドに通知
            self.stock_collector.requestCurrentPrice.emit()
            # ----------------------------------------------
        elif self.ts_start_2h <= self.ts_system <= self.ts_ca:
            # ----------------------------------------------
            # 🧿 現在価格の取得要求をスレッドに通知
            self.stock_collector.requestCurrentPrice.emit()
            # ----------------------------------------------
        elif self.ts_ca < self.ts_system:
            self.timer.stop()
            self.logger.info("タイマーを停止しました。")
            # 収集したデータの保存
            self.stock_collector.requestSaveDataFrame.emit()
        else:
            pass

        # ツールバーの時刻を更新
        self.toolbar.updateTime(self.ts_system)

    def on_save_completed(self, state:bool):
        if state:
            self.logger.info("データを正常に保存しました。")
        else:
            self.logger.info("データを正常に保存できませんでした。")

    def on_ticker_list(self, list_ticker: list, dict_name: dict):
        for ticker in list_ticker:
            print(ticker, dict_name[ticker])

    def on_stock_collector_ready(self):
        self.logger.info(f"StockCollector is ready.")
