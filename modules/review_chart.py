import logging
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
    plot_rsi,
    plot_verticals,
)
from structs.res import AppRes
from widgets.containers import Widget
from widgets.layouts import VBoxLayout


class ReviewChart(Widget):
    IMAGE_WIDTH = 680
    IMAGE_HEIGHT = 700

    def __init__(self, res: AppRes):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.res = res
        self.setMinimumSize(self.IMAGE_WIDTH, self.IMAGE_HEIGHT)

        layout = VBoxLayout()
        self.setLayout(layout)

        # Matplotlib の共通設定
        fm.fontManager.addfont(res.path_monospace)

        # FontPropertiesオブジェクト生成（名前の取得のため）
        self.font_prop = font_prop = fm.FontProperties(fname=res.path_monospace)

        plt.rcParams["font.family"] = font_prop.get_name()
        plt.rcParams["font.size"] = 9

        self.fig = fig = Figure(figsize=(self.IMAGE_WIDTH / 100., self.IMAGE_HEIGHT / 100.))
        # キャンバスを表示
        self.canvas = canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(canvas)

        self.n = n = 5
        self.ax = ax = dict()
        self.gs = gs = fig.add_gridspec(
            n, 1, wspace=0, hspace=0,
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
        # 一旦、Axes をクリア
        self.clearAxes()

        # 株価と VWAP
        i = 0
        plot_price_vwap(self.ax[i], df, title, dict_ts, dict_setting)

        # RSI
        i += 1
        plot_rsi(self.ax[i], df, dict_setting)

        # モメンタム
        i += 1
        plot_momentum(self.ax[i], df, dict_setting)

        # 含み益
        i += 1
        plot_profit(self.ax[i], df, dict_setting)

        # ドローダウン
        i += 1
        plot_drawdown(self.ax[i], df, dict_setting)

        # --- クロス・シグナル、その他縦線系 ---
        plot_verticals(self.n, self.ax, df, dict_ts, dict_setting)

        # タイト・レイアウト
        self.fig.tight_layout()

        # 再描画
        self.canvas.draw()

        # 保存だけ実行
        self.fig.savefig(name_img, dpi=100)
        self.logger.info(f"{name_img} に保存しました。")

    def getCanvas(self) -> FigureCanvas:
        return self.canvas


class ReviewChartNavigation(NavigationToolbar):
    def __init__(self, canvas: FigureCanvas):
        super().__init__(canvas)
