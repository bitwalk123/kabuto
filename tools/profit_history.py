import datetime
import os
import re

import matplotlib as mpl
import pandas as pd
from PySide6.QtCore import QMargins, Qt
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QMainWindow, QToolBar
from matplotlib import (
    dates as mdates,
    font_manager as fm,
    ticker,
)
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

from structs.res import AppRes
from widgets.dialogs import DlgCSVFileSel2


class ProfitHistoryChart(FigureCanvas):

    def __init__(self, res: AppRes):
        self.fig = Figure()
        super().__init__(self.fig)
        self.setContentsMargins(QMargins(0, 0, 0, 0))
        self.setMinimumWidth(res.profit_history_width)
        self.setFixedHeight(res.profit_history_height)

        # Font setting
        FONT_PATH = 'fonts/RictyDiminished-Regular.ttf'
        fm.fontManager.addfont(FONT_PATH)
        font_prop = fm.FontProperties(fname=FONT_PATH)
        # ★ 全体フォント適用
        mpl.rcParams['font.family'] = font_prop.get_name()
        mpl.rcParams["font.size"] = 8

        # Plot margin
        self.fig.subplots_adjust(
            left=0.10,
            right=0.99,
            top=0.9,
            bottom=0.1,
        )

        self.ax = self.fig.add_subplot(111)

    def plot(self, file_path: str):
        # === ファイルの読み込みと前処理 ===
        pattern = re.compile(r".+_([0-9]{4})([0-9]{2})([0-9]{2})\.csv")
        if m := pattern.match(file_path):
            year = m.group(1)
            month = m.group(2)
            day = m.group(3)
        else:
            raise ValueError(file_path)
        date_str = "-".join([year, month, day])
        list_head = ["注文日時", "銘柄", "売買", "約定代金[円]"]
        df = pd.read_csv(file_path, encoding="shift_jis")[list_head]
        df.index = [pd.to_datetime(f"{year}/{d}") for d in df["注文日時"]]
        df.index.name = "Datetime"
        df.sort_index(inplace=True)
        df["約定代金[円]"] = [int(s.replace(",", "")) for s in df["約定代金[円]"]]

        n = len(df)
        for r in range(0, n, 2):
            dt1 = df.index[r]
            dt2 = df.index[r + 1]
            v1 = df.loc[dt1, "約定代金[円]"]
            v2 = df.loc[dt2, "約定代金[円]"]
            position = df.loc[dt1, "売買"]
            if position == "買建":
                df.loc[dt2, "profit"] = v2 - v1
            elif position == "売建":
                df.loc[dt2, "profit"] = v1 - v2
            else:
                raise ValueError(position)

        df.dropna(subset=["profit"], inplace=True)
        df["cumsum"] = df["profit"].cumsum()
        list_dt = df.index

        dt_start = pd.to_datetime(f"{date_str} 09:00:00")
        dt_end = pd.to_datetime(f"{date_str} 15:30:00")
        profit = df["cumsum"].tail(1).iloc[0]

        df.at[dt_start, "cumsum"] = 0
        df.at[dt_end, "cumsum"] = profit
        df.sort_index(inplace=True)

        print(df)

        # === プロット ===
        self.ax.set_title(f"{date_str} : 本日の実現損益の時系列トレンド")
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        self.ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
        self.ax.set_ylabel("実現損益（円）")
        self.ax.plot(df["cumsum"], color="red", linewidth=0.75, zorder=20)

        td = datetime.timedelta(minutes=10)
        self.ax.set_xlim(dt_start - td, dt_end + td)
        self.ax.axhline(y=0, color="black", linewidth=0.5, zorder=10)

        for t in list_dt:
            self.ax.axvline(x=t, color="darkgreen", linewidth=0.5, linestyle="solid", zorder=10)

        self.ax.grid(axis="y")

        self.fig.canvas.draw_idle()

        # === 保存先の処理 ===
        dir_target = os.path.join("output", year, f"{int(month):02d}", f"{int(day):02d}")
        os.makedirs(dir_target, exist_ok=True)
        name_img = os.path.join(dir_target, f"trend_profit.png")
        self.set_save_config(name_img)


    def set_save_config(self, path_img: str):
        # ディレクトリはデータファイルと同じ
        mpl.rcParams["savefig.directory"] = os.path.dirname(path_img)
        # 保存ファイルは、拡張子以外はデータファイルと同じ
        basename = os.path.basename(path_img)
        self.get_default_filename = lambda: basename


class ProfitHistory(QMainWindow):
    def __init__(self):
        super().__init__()
        self.res = res = AppRes()

        toolbar = QToolBar()
        self.addToolBar(toolbar)
        self.csv = action_open = QAction(
            QIcon(os.path.join(res.dir_image, "csv.png")),
            "CSV ファイルを開く",
            self
        )
        action_open.triggered.connect(self.on_select_stockorder)
        toolbar.addAction(action_open)

        self.chart = chart = ProfitHistoryChart(res)
        self.setCentralWidget(chart)

        navtoolbar = NavigationToolbar(chart)
        self.addToolBar(Qt.ToolBarArea.BottomToolBarArea, navtoolbar)

    def on_select_stockorder(self):
        dlg_file = DlgCSVFileSel2(self.res)
        if dlg_file.exec():
            file_path = dlg_file.selectedFiles()[0]
        else:
            return
        self.chart.plot(file_path)
