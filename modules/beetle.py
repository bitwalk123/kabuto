import datetime
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QFileDialog

from funcs.setting import load_setting
from funcs.tse import get_ticker_name_list
from modules.kabuto import Kabuto
from modules.review_chart import ReviewChart, ReviewChartNavigation
from structs.res import AppRes
from widgets.containers import MainWindow
from widgets.statusbars import StatusBar
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

        # メイン・チャート
        self.chart = chart = ReviewChart(res)
        self.setCentralWidget(chart)

        # ステータス・バー
        statusbar = StatusBar(res)
        canvas = chart.getCanvas()
        statusbar.addWidget(ReviewChartNavigation(canvas))
        self.setStatusBar(statusbar)

    def on_open_clicked(self):
        dlg = QFileDialog()
        dlg.setNameFilters("CSV files (*.csv)")
        dlg.setOption(QFileDialog.Option.DontUseNativeDialog)
        if dlg.exec():
            filename = dlg.selectedFiles()[0]
            self.gen_review_chart(filename)
        else:
            print("Canceled!")

    def gen_review_chart(self, filename: str):
        # ファイルの読み込み
        df: pd.DataFrame = pd.read_csv(filename, index_col=0)
        df.index = pd.to_datetime(df.index)
        if m := self.pattern_code.match(filename):
            code = m.group(1)
        else:
            code = "0000"
        dict_setting: dict[str, Any] = load_setting(self.res, code)
        name: str = get_ticker_name_list([code])[code]

        # 取引時刻情報
        dt_date: datetime.date = df.index[0].date()
        t_start = datetime.time(9, 0)
        dt_start = datetime.datetime.combine(dt_date, t_start)
        t_end = datetime.time(15, 30)
        dt_end = datetime.datetime.combine(dt_date, t_end)
        dict_ts: dict[str, datetime.datetime] = {"start": dt_start, "end": dt_end}

        # プロットタイトル
        title: str = f"{dt_date}: {name} ({code})"

        # ファイルパスの定義
        path_src = Path(filename)
        # 拡張子を.csvに変更
        path_img = path_src.with_suffix('.png')

        # チャートの生成
        self.chart.draw(df, title, dict_ts, dict_setting, str(path_img))
