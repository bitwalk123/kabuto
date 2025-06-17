import os

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QMainWindow

from structs.res import AppRes


class SpotTrade(QMainWindow):
    def __init__(self, res: AppRes):
        super().__init__()
        self.res = res

        icon = QIcon(os.path.join(res.dir_image, 'pig.png'))
        self.setWindowIcon(icon)
        self.setWindowTitle("現物取引")

