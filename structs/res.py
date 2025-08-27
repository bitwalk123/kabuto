import os

import pandas as pd
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QStyle, QWidget


class AppRes:
    dir_collection = 'collection'
    dir_conf = 'conf'
    dir_excel = 'excel'
    dir_font = 'fonts'
    dir_image = 'images'
    dir_model = 'models'
    dir_output = 'output'
    dir_report = 'report'
    dir_training = 'training'
    dir_transaction = 'transaction'

    excel_collector = "collector.xlsm"
    excel_portfolio = "portfolio.xlsm"

    debug = False

    tse = 'https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls'

    path_monospace = "fonts/RictyDiminished-Regular.ttf"

    def __init__(self):
        # システムディレクトリのチェック
        list_dir = [
            self.dir_collection,
            self.dir_excel,
            self.dir_output,
            self.dir_report,
            self.dir_training,
            self.dir_transaction,
        ]
        for dirname in list_dir:
            self.check_system_dir(dirname)

    @staticmethod
    def check_system_dir(dirname: str):
        if not os.path.exists(dirname):
            os.mkdir(dirname)

    @staticmethod
    def getBuiltinIcon(parent: QWidget, name: str) -> QIcon:
        pixmap_icon = getattr(QStyle.StandardPixmap, 'SP_%s' % name)
        return parent.style().standardIcon(pixmap_icon)

    def getJPXTickerList(self) -> pd.DataFrame:
        return pd.read_excel(self.tse)


class YMD:
    year: int = 0
    month: int = 0
    day: int = 0


class HMS:
    hour: int = 0
    minute: int = 0
    second: int = 0
