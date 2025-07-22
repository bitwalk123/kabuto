import logging
import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QCloseEvent
from PySide6.QtWidgets import QMainWindow

from matisse.matisse_dock import DockMatisse
from matisse.matisse_rss import RssConnector
from structs.res import AppRes


class Matisse(QMainWindow):
    """
    MarketSPEED 2 RSS 用いた信用取引テスト
    """

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.res = res = AppRes()
        self.ticker = ""

        # 信用取引テスト用 Excel ファイル
        excel_path = 'target_test.xlsm'
        # Excel RSS と接続するスレッド・インスタンス
        self.rss_connector = rss_connector = RssConnector(res, excel_path)
        rss_connector.threadReady.connect(self.on_rss_connector_ready)
        rss_connector.worker.notifyTickerList.connect(self.on_ticker_list)
        rss_connector.worker.saveCompleted.connect(self.on_save_completed)
        rss_connector.start()

        # GUI
        icon = QIcon(os.path.join(res.dir_image, "matisse.png"))
        self.setWindowIcon(icon)
        self.setWindowTitle("信用取引テスト")

        self.dock = dock = DockMatisse(res, self.ticker)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def closeEvent(self, event: QCloseEvent):
        """
        アプリ終了イベント
        :param event:
        :return:
        """

        """
        # ---------------------------------------------------------------------
        # タイマーの停止
        # ---------------------------------------------------------------------
        if self.timer.isActive():
            self.timer.stop()
            self.logger.info("タイマーを停止しました。")
        """

        # ---------------------------------------------------------------------
        # Thread Stock Collector の削除
        # ---------------------------------------------------------------------
        if self.rss_connector.isRunning():
            self.rss_connector.requestStopProcess.emit()
            self.logger.info("Stopping StockCollector...")
            self.rss_connector.quit()  # スレッドのイベントループに終了を指示
            self.rss_connector.wait()  # スレッドが完全に終了するまで待機
            self.logger.info("StockCollector safely terminated.")

        # ---------------------------------------------------------------------
        self.logger.info(f"{__name__} stopped and closed.")
        event.accept()

    def on_rss_connector_ready(self):
        self.logger.info(f"StockCollector is ready.")

    def on_save_completed(self, state: bool):
        if state:
            self.logger.info("データを正常に保存しました。")
        else:
            self.logger.info("データを正常に保存できませんでした。")

    def on_ticker_list(self, list_ticker: list, dict_name: dict):
        """
        for ticker in list_ticker:
            print(ticker, dict_name[ticker])
        """
        # 現在のところ、ひとつのみに限定
        self.ticker = list_ticker[0]
        self.dock.setTitle(self.ticker)
        print(self.ticker, dict_name[self.ticker])
