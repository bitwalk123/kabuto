import logging
import os

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QMainWindow

from modules.toolbar import ToolBarProphet
from structs.res import AppRes


class Prophet(QMainWindow):
    __app_name__ = "Prophet"
    __version__ = "0.0.1"
    __author__ = "Fuhito Suguri"
    __license__ = "MIT"

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)  # モジュール固有のロガーを取得
        self.res = res = AppRes()

        self.setWindowIcon(QIcon(os.path.join(res.dir_image, "inference.png")))
        title_win = f"{self.__app_name__} - {self.__version__}"
        self.setWindowTitle(title_win)

        toolbar = ToolBarProphet(res)
        toolbar.clickedPlay.connect(self.on_start)
        self.addToolBar(toolbar)

    def on_start(self):
        pass