import sys

from PySide6.QtWidgets import QMainWindow

from structs.res import AppRes

if sys.platform == "win32":
    debug = False
else:
    # Windows でないプラットフォーム上ではデバッグ・モードになる
    debug = True


class Rhino(QMainWindow):
    __app_name__ = "Rhino"
    __version__ = "0.9.0"
    __author__ = "Fuhito Suguri"
    __license__ = "MIT"

    def __init__(self, options: list = None):
        super().__init__()
        global debug  # グローバル変数であることを明示
        self.res = res = AppRes()
