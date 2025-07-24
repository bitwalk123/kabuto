import logging
import os
import sys

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QMainWindow

from structs.res import AppRes

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

        # コンソールから起動した際のオプション・チェック
        if len(options) > 0:
            for option in options:
                if option == "debug":
                    debug = True  # Windows 上でデバッグ・モードを使用する場合
        # デバッグ・モードを保持
        res.debug = debug

        # ウィンドウ・タイトル文字列
        title_win = f"{self.__app_name__} - {self.__version__}"

        #######################################################################
        # NORMAL / DBUG モード固有の設定
        if debug:
            self.logger.info(f"{__name__} executed as DEBUG mode!")
            # ウィンドウ・タイトル（デバッグモード）文字列
            title_win = f"{title_win} [debug mode]"
            self.timer_interval = 100  # タイマー間隔（ミリ秒）（デバッグ時）
        else:
            self.logger.info(f"{__name__} executed as NORMAL mode!")
            self.timer_interval = 1000  # タイマー間隔（ミリ秒）
        #
        #######################################################################

        # ウィンドウアイコンとタイトルを設定
        icon = QIcon(os.path.join(res.dir_image, "rhino.png"))
        self.setWindowIcon(icon)
        self.setWindowTitle(title_win)
