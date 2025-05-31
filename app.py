import logging
import os
import sys

from PySide6.QtCore import QThread, QTimer, Signal
from PySide6.QtGui import QIcon, QCloseEvent
from PySide6.QtWidgets import QApplication, QMainWindow

from funcs.logs import setup_logging
from funcs.uis import clear_boxlayout
from modules.trader_pyqtgraph import Trader
from modules.xlreviewer import ReviewWorker
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
    request_reviewer_init = Signal()

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
            # タイマー間隔（ミリ秒）
            self.t_interval = 100
        else:
            self.logger.info(f"{__name__} executed as NORMAL mode!")
            # ウィンドウ・タイトル文字列
            title_window = f"{self.__app_name__} - {self.__version__}"
            # タイマー間隔（ミリ秒）
            self.t_interval = 1000

        # デバッグ・モードを保持
        res.debug = debug

        # Excel レビュー用インスタンス（スレッド）
        self.reviewer_thread: QThread | None = None
        self.reviewer: ReviewWorker | None = None

        # ticker インスタンスを保持するリスト
        self.list_trader = list_trader = list()

        # ウィンドウアイコンとタイトルを設定
        icon = QIcon(os.path.join(res.dir_image, "kabuto.png"))
        self.setWindowIcon(icon)
        self.setWindowTitle(title_window)

        # ツールバー
        toolbar = ToolBar(res)
        toolbar.excelSelected.connect(self.on_create_reviewer_thread)
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
        timer.timeout.connect(self.on_update_data)
        timer.setInterval(self.t_interval)

    def closeEvent(self, event: QCloseEvent):
        if self.timer.isActive():
            self.timer.stop()

        if self.reviewer_thread is not None:
            try:
                if self.reviewer_thread.isRunning():
                    self.reviewer_thread.quit()
                    self.reviewer_thread.deleteLater()
                    self.logger.info(f"reviewer スレッドを削除しました。")
            except RuntimeError as e:
                self.logger.info(f"終了時: {e}")

        self.logger.info(f"{__name__} stopped and closed.")
        event.accept()

    def on_create_reviewer_thread(self, excel_path: str):
        """
        保存したティックデータをレビューするためのワーカースレッドを作成

        このスレッドは QThread の　run メソッドを継承していないので、
        明示的にワーカースレッドを終了する処理をしない限り残っていてイベント待機状態になっている。

        :param excel_path:
        :return:
        """
        # Excelを読み込むスレッド処理
        self.reviewer_thread = reviewer_thread = QThread()
        self.reviewer = reviewer = ReviewWorker(excel_path)
        reviewer.moveToThread(reviewer_thread)

        # QThread が開始されたら、ワーカースレッド内で初期化処理を開始するシグナルを発行
        reviewer_thread.started.connect(self.request_reviewer_init.emit)
        self.request_reviewer_init.connect(reviewer.loadExcel)

        # シグナルとスロットの接続
        reviewer.notifyTickerN.connect(self.on_create_trader)
        reviewer.threadFinished.connect(self.on_thread_finished)
        reviewer.threadFinished.connect(reviewer_thread.quit)  # スレッド終了時
        reviewer_thread.finished.connect(reviewer_thread.deleteLater)  # スレッドオブジェクトの削除

        # スレッドを開始
        self.reviewer_thread.start()

    def on_create_trader(self, list_ticker: list, dict_times: dict):
        # 配置済みの Trader インスタンスを消去
        clear_boxlayout(self.layout)

        # Trader リストのクリア
        self.list_trader = list()

        # Trader の配置
        for ticker in list_ticker:
            trader = Trader(self.res)
            trader.setTitle(ticker)
            trader.setTimeRange(*dict_times[ticker])
            self.layout.addWidget(trader)
            self.list_trader.append(trader)

    def on_thread_finished(self, result: bool):
        if result:
            print("スレッドが正常終了しました。")
        else:
            print("スレッドが異常終了しました。")

        if self.timer.isActive():
            self.timer.stop()

    def on_update_data(self):
        pass


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
