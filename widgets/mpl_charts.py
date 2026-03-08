import pandas as pd
from PySide6.QtWidgets import QSizePolicy
from matplotlib import (
    dates as mdates,
    font_manager as fm,
    pyplot as plt,
    ticker,
)
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

from structs.res import AppRes
from widgets.containers import ScrollArea, Widget
from widgets.layouts import VBoxLayout


class Chart(Widget):
    """
    チャート用 FigureCanvas の雛形
    """

    def __init__(self, res: AppRes):
        super().__init__()
        # フォント設定
        fm.fontManager.addfont(res.path_monospace)
        font_prop = fm.FontProperties(fname=res.path_monospace)
        plt.rcParams["font.family"] = font_prop.get_name()
        plt.rcParams["font.size"] = 11
        # ダークモードの設定
        # plt.style.use("dark_background")

        # Figure オブジェクト
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        self.canvas.updateGeometry()

        # レイアウトに追加
        layout = VBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # 軸領域
        self.ax = self.fig.add_subplot(111)


class MplChart(FigureCanvas):
    def __init__(self):
        self.fig = Figure()
        super().__init__(self.fig)

        # Font setting
        FONT_PATH = 'fonts/RictyDiminished-Regular.ttf'
        fm.fontManager.addfont(FONT_PATH)
        font_prop = fm.FontProperties(fname=FONT_PATH)
        plt.rcParams['font.family'] = font_prop.get_name()
        plt.rcParams['font.size'] = 14

        # dark mode
        # plt.style.use('dark_background')

        # Plot margin
        self.fig.subplots_adjust(
            left=0.075,
            right=0.98,
            top=0.95,
            bottom=0.05,
        )

        # Axes
        self.ax = dict()

    def clearAxes(self):
        axs = self.fig.axes
        for ax in axs:
            ax.cla()
            ax.grid()

    def initAxes(self, ax, n: int):
        if n > 1:
            gs = self.fig.add_gridspec(
                n, 1,
                wspace=0.0, hspace=0.0,
                height_ratios=[3 if i == 0 else 1 for i in range(n)]
            )
            for i, axis in enumerate(gs.subplots(sharex='col')):
                ax[i] = axis
                ax[i].grid()
        else:
            ax[0] = self.fig.add_subplot()
            ax[0].grid()

    def initChart(self, n: int):
        self.removeAxes()
        self.initAxes(self.ax, n)

    def refreshDraw(self):
        self.fig.canvas.draw()

    def removeAxes(self):
        for ax in list(self.fig.axes):
            ax.remove()


class ChartNavigation(NavigationToolbar):
    def __init__(self, canvas: FigureCanvas):
        super().__init__(canvas)


def get_param_string(dict_param: dict) -> str:
    param = (
        f"PERIOD_MA_1 = {dict_param['PERIOD_MA_1']} / "
        f"PERIOD_MA_2 = {dict_param['PERIOD_MA_2']} / "
        f"PERIOD_MR = {dict_param['PERIOD_MR']} / "
        f"THRESHOLD_MR = {dict_param['THRESHOLD_MR']}"
    )
    return param


class ObsChart(ScrollArea):
    """
    観測値チャート用
    """

    def __init__(self, res: AppRes):
        super().__init__()
        # フォント設定
        fm.fontManager.addfont(res.path_monospace)
        font_prop = fm.FontProperties(fname=res.path_monospace)
        plt.rcParams["font.family"] = font_prop.get_name()
        plt.rcParams["font.size"] = 9
        # ダークモードの設定
        # plt.style.use("dark_background")

        # Figure オブジェクト
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        self.canvas.updateGeometry()
        self.setWidget(self.canvas)

        # 余白設定
        self.fig.subplots_adjust(
            left=0.05,
            right=0.99,
            top=0.953,
            bottom=0.06,
        )

    def removeAxes(self):
        """
        axs = list(self.fig.axes) が「安全」な理由は：
        - ax.remove() が self.fig.axes を変更する
        - 元リストを走査しながら変更するとバグ・例外・不整合が起きる
        - コピーしたリストを使えば、ループ中に元リストが変わっても問題ない
        - 全ての Axes を確実に削除できる

        つまり、「元のリストを変更しながらループしない」ための防御策です。
        :return:
        """
        for ax in list(self.fig.axes):
            ax.remove()

    def updateData(self, df: pd.DataFrame, dict_param: dict, title: str = ""):
        self.removeAxes()
        list_height_ratio = list()
        list_col = list(df.columns)
        # MA2 は MA1 と重ねてプロットするので除外です。
        if "MA2" in list_col:
            idx_ma_2 = list_col.index("MA2")
            list_col.pop(idx_ma_2)
        n = len(list_col)
        for colname in list_col:
            if colname in ["Price", "MA1"]:
                list_height_ratio.append(2)
            elif colname in ["低ボラ", "建玉", "損益M", "ロス", "利確"]:
                list_height_ratio.append(0.5)
            else:
                list_height_ratio.append(1)
        gs = self.fig.add_gridspec(
            n, 1,
            wspace=0.0, hspace=0.0,
            height_ratios=list_height_ratio
        )
        ax = dict()
        for i, axis in enumerate(gs.subplots(sharex="col")):
            ax[i] = axis
            ax[i].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax[i].grid()

        for i, colname in enumerate(list_col):
            if colname == "Price":
                ax[i].plot(df[colname], linewidth=0.5)
                ax[i].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
                ax[i].set_ylabel("株価")
            elif colname == "MA1":
                ax[i].plot(df[colname], linewidth=0.5, color="magenta")
                ax[i].plot(df["MA2"], linewidth=0.5, color="darkgreen")
                ax[i].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
                ax[i].set_ylabel("株価（移動平均）")
            elif colname == "低ボラ":
                x = df.index
                y = df[colname]
                ax[i].fill_between(x, 0, y, where=y == 1.0, color='green', alpha=0.4, interpolate=True)
                ax[i].set_ylabel(colname)
            elif colname == "建玉":
                x = df.index
                y = df[colname]
                ax[i].fill_between(x, 0, y, where=y > 0.0, color='green', alpha=0.4, interpolate=True)
                ax[i].fill_between(x, 0, y, where=y < 0.0, color='magenta', alpha=0.4, interpolate=True)
                ax[i].set_ylim(-1.1, 1.1)
                ax[i].set_ylabel(colname)
            elif colname == "損益M":
                ax[i].plot(df[colname], linewidth=0.5)
                _, y_max = ax[i].get_ylim()
                ax[i].set_ylim(-0.1, y_max)
                ax[i].set_ylabel(colname)
            else:
                ax[i].plot(df[colname], linewidth=0.5)
                ax[i].set_ylabel(colname)

        # チャートのタイトル
        ax[0].set_title(title)

        # パラメータ情報
        self.fig.suptitle(get_param_string(dict_param), fontsize=7)

        # 再描画
        self.canvas.setFixedHeight(len(list_height_ratio) * 100)
        self.canvas.updateGeometry()
        self.canvas.draw()


class TrendChart(Widget):
    """
    リアルタイム用トレンドチャート
    """

    def __init__(self, res: AppRes):
        super().__init__()
        # ウィンドウのサイズ制約
        self.setMinimumWidth(1000)
        self.setFixedHeight(300)

        # フォント設定
        fm.fontManager.addfont(res.path_monospace)
        font_prop = fm.FontProperties(fname=res.path_monospace)
        plt.rcParams["font.family"] = font_prop.get_name()
        plt.rcParams["font.size"] = 12
        # ダークモードの設定
        plt.style.use("dark_background")

        # Figure オブジェクト
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        # 余白設定
        self.fig.subplots_adjust(left=0.075, right=0.99, top=0.9, bottom=0.08)
        self.canvas.updateGeometry()

        # レイアウトに追加
        layout = VBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.ax = self.fig.add_subplot(111)
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        self.ax.grid(True, lw=0.5)

    def reDraw(self):
        # データ範囲を再計算
        self.ax.relim()
        # y軸のみオートスケール
        self.ax.autoscale_view(scalex=False, scaley=True)  # X軸は固定、Y軸は自動
        # 再描画
        self.canvas.draw_idle()

    def setTitle(self, title: str):
        self.ax.set_title(title)
