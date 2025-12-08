import pandas as pd
from PySide6.QtWidgets import QSizePolicy
from matplotlib import (
    dates as mdates,
    font_manager as fm,
    pyplot as plt,
    ticker,
)
from matplotlib.backends.backend_qtagg import (
    NavigationToolbar2QT as NavigationToolbar,
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.figure import Figure

from funcs.technical import calc_ma, calc_mr
from structs.res import AppRes
from widgets.containers import Widget, ScrollArea
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
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
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
        self.ax = self.figure.add_subplot(111)


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
    def __init__(self, chart: FigureCanvas):
        super().__init__(chart)


class TickChart(Chart):
    """
    ティックチャート用
    """

    def __init__(self, res: AppRes):
        super().__init__(res)
        self.ax2 = self.ax.twinx()
        # 余白設定
        self.figure.subplots_adjust(
            left=0.05,
            right=0.96,
            top=0.94,
            bottom=0.06,
        )
        plt.rcParams['font.size'] = 14
        self.space = "            "

        # タイムスタンプへ時差を加算用（Asia/Tokyo)
        self.tz = 9. * 60 * 60

        self.removeAxes()

    def removeAxes(self):
        self.ax.remove()
        self.ax2.remove()
        self.ax = self.figure.add_subplot(111)
        self.ax2 = self.ax.twinx()
        self.ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%H:%M")
        )

    def updateData(self, df: pd.DataFrame, dict_param: dict, title: str):
        # トレンドライン（株価と指標）
        df.index = [pd.Timestamp(ts + self.tz, unit='s') for ts in df["Time"]]
        period_mad_1 = dict_param["PERIOD_MA_1"]
        period_mad_2 = dict_param["PERIOD_MA_2"]
        period_mr = dict_param["PERIOD_MR"]
        threshold_miqr = dict_param["THRESHOLD_MR"]
        colname_ma_1, colname_ma_2 = calc_ma(df, period_mad_1, period_mad_2)
        colname_mr = calc_mr(df, period_mr)

        # プロット用データ
        ser_price = df["Price"]
        ser_ma_1 = df[colname_ma_1]
        ser_ma_2 = df[colname_ma_2]
        ser_miqr = df[colname_mr]

        # 消去
        self.removeAxes()

        # プロット (y)
        lns1 = self.ax.plot(ser_price, color="lightgray", linewidth=0.5, linestyle="solid", label="Price")
        lns2 = self.ax.plot(ser_ma_1, linewidth=1, linestyle="solid", label=colname_ma_1)
        lns3 = self.ax.plot(ser_ma_2, linewidth=1, linestyle="solid", label=colname_ma_2)

        self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        self.ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
        self.ax.grid(True, lw=0.5)
        self.ax.set_title(title)
        # y 軸調整
        y_min, y_max = self.ax.get_ylim()
        y_min_2 = 2 * y_min - y_max
        if y_min_2 < 0:
            y_min_2 = 0
        self.ax.set_ylim(y_min_2, y_max)
        yticks = self.ax.get_yticklabels()
        idx_min = len(yticks) // 2 - 1
        for i in range(idx_min):
            yticks[i].set_visible(False)
        self.ax.set_ylabel(f"{self.space}Price [JPY]")

        # プロット (y2)
        lns4 = self.ax2.plot(ser_miqr, color="C2", linewidth=0.75, linestyle="solid", label=colname_mr)
        x = ser_miqr.index
        y = ser_miqr.values
        self.ax2.fill_between(
            x, 0, 1,
            where=y < threshold_miqr,
            color='black',
            alpha=0.05,
            transform=self.ax2.get_xaxis_transform()
        )
        # y2 軸調整
        y_min, y_max = self.ax2.get_ylim()
        y_min = 0.0
        y_max_2 = 2 * y_max - y_min
        self.ax2.set_ylim(y_min, y_max_2)
        y2ticks = self.ax2.get_yticklabels()
        idx_max = len(y2ticks) // 2 + 2
        for i in range(idx_max, len(y2ticks)):
            y2ticks[i].set_visible(False)
        self.ax2.yaxis.set_label_position("right")
        self.ax2.set_ylabel(f"{colname_mr}{self.space}")

        # added these three lines
        lns = lns1 + lns2 + lns3 + lns4
        labs = [l.get_label() for l in lns]
        self.ax.legend(lns, labs, bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.25, fontsize=7)
        # self.ax.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.5, fontsize=7)

        # 再描画
        self.canvas.draw()


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

        # 軸領域
        self.ax = self.fig.add_subplot(111)

        # 余白設定
        self.fig.subplots_adjust(
            left=0.05,
            right=0.99,
            top=0.95,
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

    def updateData(self, df: pd.DataFrame, title: str = ""):
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

        # 再描画
        self.canvas.setFixedHeight(len(list_height_ratio) * 100)
        self.canvas.updateGeometry()
        self.canvas.draw()


class TrendChart(FigureCanvas):
    """
    リアルタイム用トレンドチャート
    """

    def __init__(self, res: AppRes):
        # フォント設定
        fm.fontManager.addfont(res.path_monospace)
        font_prop = fm.FontProperties(fname=res.path_monospace)
        plt.rcParams["font.family"] = font_prop.get_name()
        plt.rcParams["font.size"] = 12
        # ダークモードの設定
        plt.style.use("dark_background")

        # Figure オブジェクト
        self.figure = Figure()
        # 余白設定
        self.figure.subplots_adjust(left=0.075, right=0.99, top=0.9, bottom=0.08)
        super().__init__(self.figure)

        self.ax = self.figure.add_subplot(111)
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        self.ax.grid(True, lw=0.5)

    def reDraw(self):
        # データ範囲を再計算
        self.ax.relim()
        # y軸のみオートスケール
        self.ax.autoscale_view(scalex=False, scaley=True)  # X軸は固定、Y軸は自動
        # 再描画
        self.draw()

    def setTitle(self, title: str):
        self.ax.set_title(title)
