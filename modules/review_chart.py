from typing import Any

import pandas as pd
from PySide6.QtWidgets import QSizePolicy
from matplotlib import font_manager as fm, pyplot as plt
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

from funcs.plot import (
    plot_drawdown,
    plot_momentum,
    plot_price_vwap,
    plot_profit,
    plot_verticals,
)
from structs.res import AppRes
from widgets.containers import Widget
from widgets.layouts import VBoxLayout


class ReviewChart(Widget):
    IMAGE_WIDTH = 680
    IMAGE_HEIGHT = 600

    def __init__(self, res: AppRes):
        super().__init__()
        self.res = res
        self.setMinimumSize(self.IMAGE_WIDTH, self.IMAGE_HEIGHT)

        layout = VBoxLayout()
        self.setLayout(layout)

        # Matplotlib の共通設定
        fm.fontManager.addfont(res.path_monospace)

        # FontPropertiesオブジェクト生成（名前の取得のため）
        font_prop = fm.FontProperties(fname=res.path_monospace)
        font_prop.get_name()

        plt.rcParams["font.family"] = font_prop.get_name()
        plt.rcParams["font.size"] = 9

        self.fig = fig = Figure(figsize=(self.IMAGE_WIDTH / 100., self.IMAGE_HEIGHT / 100.))
        # キャンバスを表示
        self.canvas = canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(canvas)

        self.n = n = 4
        self.ax = ax = dict()
        gs = fig.add_gridspec(
            n, 1, wspace=0.0, hspace=0.0,
            height_ratios=[1.5 if i == 0 else 1 for i in range(n)],
        )
        for i, axis in enumerate(gs.subplots(sharex="col")):
            ax[i] = axis
            ax[i].grid(axis="y")

    def clearAxes(self):
        axs = self.fig.axes
        for ax in axs:
            ax.cla()
            ax.grid(axis="y")


    def draw(
            self,
            df: pd.DataFrame,
            title: str,
            dict_ts: dict[str, Any],
            dict_setting: dict[str, Any],
            name_img: str
    ) -> None:
        self.clearAxes()

        # 1. 株価と VWAP
        plot_price_vwap(self.ax[0], df, title, dict_ts)

        # 2. モメンタム
        plot_momentum(self.ax[1], df, dict_setting)

        # 3. 含み益
        plot_profit(self.ax[2], df, dict_setting)

        # 4. ドローダウン
        plot_drawdown(self.ax[3], df, dict_setting)

        # --- クロス・シグナル、その他縦線系 ---
        plot_verticals(self.n, self.ax, df, dict_ts)

        self.fig.tight_layout()
        self.canvas.draw()

        # 保存だけ実行
        self.fig.savefig(name_img, dpi=100)
        print(f"{name_img} に保存しました。")

    def getCanvas(self) -> FigureCanvas:
        return self.canvas

class ReviewChartNavigation(NavigationToolbar):
    def __init__(self, canvas: FigureCanvas):
        super().__init__(canvas)
