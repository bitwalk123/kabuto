import logging
import os
import re
import sys
import time

from PySide6.QtCore import QThread
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QMainWindow

from funcs.logs import setup_logging
from modules.xlloader import ExcelLoader
from structs.res import AppRes
from widgets.containers import Widget
from widgets.layouts import VBoxLayout
from widgets.statusbar import StatusBar
from widgets.toolbar import ToolBar

if sys.platform == "win32":
    import xlwings as xw
    from pywintypes import com_error  # Windows 固有のライブラリ

    debug = False
else:
    debug = True


class Kabuto(QMainWindow):
    __app_name__ = "Kabuto"
    __version__ = "0.1.0"

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
        else:
            self.logger.info(f"{__name__} executed as NORMAL mode!")
            # ウィンドウ・タイトル文字列
            title_window = f"{self.__app_name__} - {self.__version__}"

        # デバッグ・モードを保持
        res.debug = debug

        # ウィンドウアイコンとタイトルを設定
        icon = QIcon(os.path.join(res.dir_image, "kabuto.png"))
        self.setWindowIcon(icon)
        self.setWindowTitle(title_window)

        # ツールバー
        toolbar = ToolBar(res)
        toolbar.excelSelected.connect(self.on_load_excel)
        self.addToolBar(toolbar)

        # メインウィジェット
        base = Widget()
        self.setCentralWidget(base)

        layout = VBoxLayout()
        base.setLayout(layout)

        # ステータス・バー
        self.statusbar = statusbar = StatusBar(res)
        self.setStatusBar(statusbar)

    def on_error(self):
        print("Excelファイルの読み込みに失敗しました。")

    def on_load_excel(self, excel_path: str):
        # Excel を読み込むスレッド処理
        self.xl_thread = QThread()
        self.xl_loader = ExcelLoader(excel_path)
        self.xl_loader.moveToThread(self.xl_thread)

        # シグナルとスロットの接続
        self.xl_thread.started.connect(self.xl_loader.run)
        self.xl_loader.threadFinished.connect(self.on_finished_loading)
        self.xl_loader.errorOccurred.connect(self.on_error)
        self.xl_loader.threadFinished.connect(self.xl_thread.quit)  # 処理完了時にスレッドを終了
        self.xl_loader.errorOccurred.connect(self.xl_thread.quit)  # エラー時にもスレッドを終了
        self.xl_thread.finished.connect(self.xl_thread.deleteLater)  # スレッドオブジェクトの削除

        # スレッドを開始
        self.xl_thread.start()

    def on_finished_loading(self, result: bool):
        if result:
            print("スレッドが正常終了しました。")
        else:
            print("スレッドが異常終了しました。")


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
