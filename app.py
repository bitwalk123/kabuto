import logging
import os
import re
import sys
import time

from PySide6.QtWidgets import QApplication, QMainWindow

from funcs.logs import setup_logging
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
            self.setWindowTitle(f"{self.__app_name__} [debug mode]")
            res.debug = debug
        else:
            self.logger.info(f"{__name__} executed as NORMAL mode!")
            self.setWindowTitle(self.__app_name__)

        toolbar = ToolBar(res)
        self.addToolBar(toolbar)

        base = Widget()
        self.setCentralWidget(base)

        layout = VBoxLayout()
        base.setLayout(layout)

        self.statusbar = statusbar = StatusBar(res)
        self.setStatusBar(statusbar)


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
