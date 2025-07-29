from matplotlib import (
    dates as mdates,
    font_manager as fm,
    pyplot as plt,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from structs.res import AppRes


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
