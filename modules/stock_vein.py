import logging
import os

from PySide6.QtGui import QIcon, QCloseEvent
from PySide6.QtWidgets import QMainWindow

from modules.stock_collector import StockCollector
from structs.res import AppRes
from widgets.toolbar import ToolBarVein


class StockVein(QMainWindow):
    def __init__(self):
        super().__init__()
        # モジュール固有のロガーを取得
        self.logger = logging.getLogger(__name__)

        self.res = res = AppRes()

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
        stock_collector.start()

    def closeEvent(self, event: QCloseEvent):
        """
        アプリ終了イベント
        :param event:
        :return:
        """
        # ---------------------------------------------------------------------
        # Thread Stock Collector の削除
        # ---------------------------------------------------------------------
        if self.stock_collector.isRunning():
            self.logger.info(f"Stopping StockCollector...")
            self.stock_collector.quit()  # スレッドのイベントループに終了を指示
            self.stock_collector.wait()  # スレッドが完全に終了するまで待機
            self.logger.info(f"StockCollector safely terminated.")

        # ---------------------------------------------------------------------
        self.logger.info(f"{__name__} stopped and closed.")
        event.accept()

    def on_ticker_list(self, list_ticker: list, dict_name: dict):
        for ticker in list_ticker:
            print(ticker, dict_name[ticker])

    def on_stock_collector_ready(self):
        self.logger.info(f"StockCollector is ready.")
