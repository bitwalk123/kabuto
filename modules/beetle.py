import os
import re

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QFileDialog

from modules.kabuto import Kabuto
from structs.res import AppRes
from widgets.containers import MainWindow
from widgets.toolbars import ToolBarBeetle


class Beetle(MainWindow):
    __app_name__ = "Beetle"
    __version__ = Kabuto.__version__
    __author__ = "Fuhito Suguri"
    __license__ = "MIT"

    def __init__(self):
        super().__init__()
        self.res = res = AppRes()
        self.pattern_code = re.compile(r".*([0-9A-X]{4})_.+\.csv")

        # ウィンドウアイコンとタイトルを設定
        self.setWindowIcon(QIcon(os.path.join(self.res.dir_image, "beetle.png")))
        title_win = f"{self.__app_name__} - {self.__version__}"
        self.setWindowTitle(title_win)

        # ツール・バー
        toolbar = ToolBarBeetle(res)
        toolbar.clickedOpen.connect(self.on_open_clicked)
        self.addToolBar(toolbar)


    def on_open_clicked(self):
        dlg = QFileDialog()
        dlg.setNameFilters(["CSV files (*.csv)"])
        dlg.setOption(QFileDialog.Option.DontUseNativeDialog)
        if dlg.exec():
            filename = dlg.selectedFiles()[0]
            print(filename)
        else:
            print("Canceled!")

