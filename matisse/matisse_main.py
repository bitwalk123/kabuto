import logging
import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QMainWindow

from matisse.matisse_dock import DockMatisse
from structs.res import AppRes


class Matisse(QMainWindow):
    """
    MarketSPEED 2 RSS 用いた信用取引テスト
    """

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.res = res = AppRes()

        # 信用取引テスト用 Excel ファイル
        self.excel_path = 'target_test.xlsm'

        # GUI
        icon = QIcon(os.path.join(res.dir_image, "matisse.png"))
        self.setWindowIcon(icon)
        self.setWindowTitle("信用取引テスト")

        ticker = "ABCD"
        self.dock = dock = DockMatisse(res, ticker)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
