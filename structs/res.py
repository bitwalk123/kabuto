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
    dir_output = 'output'
    dir_report = 'report'
    dir_transaction = 'transaction'

    excel_collector = "collector.xlsx"
    excel_portfolio = "portfolio.xlsm"

    debug = False

    tse = 'https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls'

    def __init__(self):
        if not os.path.exists(self.dir_collection):
            os.mkdir(self.dir_collection)

        if not os.path.exists(self.dir_excel):
            os.mkdir(self.dir_excel)

        if not os.path.exists(self.dir_output):
            os.mkdir(self.dir_output)

        if not os.path.exists(self.dir_report):
            os.mkdir(self.dir_report)

        if not os.path.exists(self.dir_transaction):
            os.mkdir(self.dir_transaction)

    def getBuiltinIcon(self, parent: QWidget, name: str) -> QIcon:
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
