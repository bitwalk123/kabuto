import datetime
import re
import sys
from typing import Any

from matplotlib import (
    font_manager as fm,
    pyplot as plt,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QStyle,
    QToolBar,
    QToolButton, QSizePolicy, QWidget, QVBoxLayout,
)

from funcs.plot import plot_price_vwap, plot_profit, plot_drawdown, plot_verticals
from funcs.setting import load_setting
from funcs.tse import get_ticker_name_list
from structs.res import AppRes


class PlotReview(QMainWindow):
    def __init__(self):
        super().__init__()
        self.res = AppRes()
        self.pattern_code = re.compile(r".*([0-9A-X]{4})_.+\.csv")

        toolbar = QToolBar()
        self.addToolBar(toolbar)

        but_open = QToolButton()
        but_open.setText('Open')
        but_open.setToolTip('Open file')
        but_open.setIcon(
            self.style().standardIcon(
                QStyle.StandardPixmap.SP_DirOpenIcon
            )
        )
        but_open.clicked.connect(self.on_open_clicked)
        toolbar.addWidget(but_open)

        base = QWidget()
        base.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        base.setMinimumSize(900, 400)
        self.setCentralWidget(base)
        self.layout = layout = QVBoxLayout()
        base.setLayout(layout)

    def on_open_clicked(self):
        dlg = QFileDialog()
        dlg.setOption(QFileDialog.Option.DontUseNativeDialog)
        if dlg.exec():
            filename = dlg.selectedFiles()[0]
            self.gen_review_chart(filename)
        else:
            print("Canceled!")

    def gen_review_chart(self, filename: str):
        df: pd.DataFrame = pd.read_csv(filename, index_col=0)
        df.index = pd.to_datetime(df.index)
        if m := self.pattern_code.match(filename):
            code = m.group(1)
        else:
            code = "0000"
        dict_setting: dict[str, Any] = load_setting(self.res, code)
        name: str = get_ticker_name_list([code])[code]

        dt_date: datetime.date = df.index[0].date()
        t_start = datetime.time(9, 0)
        dt_start = datetime.datetime.combine(dt_date, t_start)
        t_end = datetime.time(15, 30)
        dt_end = datetime.datetime.combine(dt_date, t_end)
        dict_ts: dict[str, datetime.datetime] = {"start": dt_start, "end": dt_end}

        title: str = f"{dt_date}: {name} ({code})"

        # Matplotlib の共通設定
        FONT_PATH = "fonts/RictyDiminished-Regular.ttf"
        fm.fontManager.addfont(FONT_PATH)

        # FontPropertiesオブジェクト生成（名前の取得のため）
        font_prop = fm.FontProperties(fname=FONT_PATH)
        font_prop.get_name()

        plt.rcParams["font.family"] = font_prop.get_name()
        plt.rcParams["font.size"] = 9

        fig = Figure(figsize=(6.8, 6))
        canvas = FigureCanvas(fig)  # 描画に必要

        n = 3
        ax = dict()
        gs = fig.add_gridspec(
            n,
            1,
            wspace=0.0,
            hspace=0.0,
            height_ratios=[1.5 if i == 0 else 1 for i in range(n)],
        )
        for i, axis in enumerate(gs.subplots(sharex="col")):
            ax[i] = axis
            ax[i].grid(axis="y")

        # 株価と VWAP
        plot_price_vwap(ax[0], df, title, dict_ts)

        # 含み益
        plot_profit(ax[1], df, dict_setting)

        # ドローダウン
        ax2 = ax[2]
        plot_drawdown(ax2, df, dict_setting)

        # クロス・シグナル、その他縦線系
        plot_verticals(n, ax, df, dict_ts)

        fig.tight_layout()

        # 画面に表示（layout.addWidget）せずに保存だけ実行
        output = "temp.png"
        fig.savefig(output)
        print(f"{output} に保存しました。")

        # キャンバスを表示
        self.layout.addWidget(canvas)


def main():
    # QApplication は sys.argv を処理するので、そのまま引数を渡すのが一般的。
    app = QApplication(sys.argv)

    win = PlotReview()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
