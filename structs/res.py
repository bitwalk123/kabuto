import os

import pandas as pd
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QStyle, QWidget


# from widgets.misc import TickFont


class AppRes:
    # トレンドチャートの幅、高さ
    trend_width: int = 1500
    trend_height: int = 250
    trend_n_max: int = 3  # ビューに表示できるチャートの数

    # code_default = "7011"  # デフォルトの銘柄コード
    # code_default: str = "8306"  # デフォルトの銘柄コード
    code_default: str = "9984"  # デフォルトの銘柄コード

    dir_collection: str = "collection"
    dir_conf: str = "conf"
    url_conf: str = "https://192.168.0.36/~bitwalk/conf"
    dir_doe: str = "doe"
    dir_excel: str = "excel"
    dir_font: str = "fonts"
    dir_image: str = "images"
    dir_info: str = "info"
    dir_log: str = "logs"
    dir_model: str = "models"
    dir_output: str = "output"
    dir_report: str = "report"
    dir_temp: str = "tmp"
    dir_training: str = "training"
    dir_transaction: str = "transaction"

    ssh_key_path: str = "~/.ssh/id_rsa"
    remote_user: str = "bitwalk"
    remote_host: str = "192.168.0.36"
    remote_conf_dir: str = "/home/bitwalk/public_html/conf/"

    excel_collector: str = "collector.xlsm"
    excel_portfolio: str = "portfolio.xlsm"

    debug: bool = False

    tse: str = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"

    path_monospace: str = "fonts/RictyDiminished-Regular.ttf"
    name_tick_font: str | None = None

    def __init__(self):
        # システムディレクトリのチェック
        list_dir = [
            self.dir_collection,
            self.dir_excel,
            self.dir_info,
            self.dir_log,
            self.dir_model,
            self.dir_output,
            self.dir_report,
            self.dir_temp,
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
        pixmap_icon = getattr(QStyle.StandardPixmap, "SP_%s" % name)
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
