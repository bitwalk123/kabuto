import logging
import sys

from PySide6.QtWidgets import QMainWindow

from structs.res import AppRes

if sys.platform == "win32":
    debug = False
else:
    debug = True  # Windows 以外ではデバッグ・モード


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
