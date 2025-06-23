import sys
import random
import time

from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PySide6.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import datetime  # 日付時刻処理のためにインポート


class TrendChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.ax = self.figure.add_subplot(111)
        self.ax.grid(True)
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Value")
        self.ax.set_title("Real-time Trend Chart")

        self.list_x = list()
        self.list_y = list()

        # プロットラインの初期化
        self.line, = self.ax.plot(
            self.list_x, self.list_y, label='Trend', color='cyan'
        )

        # 最新の点を表示するためのプロットを初期化
        self.latest_point, = self.ax.plot(
            [], [],
            marker='o', markersize=8, color='red',
            label='Latest Data'
        )

        self.ax.legend()

        # X軸の初期表示範囲を設定するための変数
        self._x_min = None
        self._x_max = None

    def appendData(self, x, y):
        # データを追加（ここでは古いデータの削除は行わない）
        self.list_x.append(x)
        self.list_y.append(y)

        # グラフを更新
        self.line.set_xdata(self.list_x)
        self.line.set_ydata(self.list_y)

        # 最新の点を更新
        if self.list_x:
            self.latest_point.set_xdata([self.list_x[-1]])
            self.latest_point.set_ydata([self.list_y[-1]])
        else:
            self.latest_point.set_xdata([])
            self.latest_point.set_ydata([])

        # X軸の範囲は固定（set_x_rangeで設定されていればそれを使用）
        # Y軸はデータに基づいて自動調整
        self.ax.relim()
        self.ax.autoscale_view(scalex=False, scaley=True)  # X軸は固定、Y軸は自動

        # Canvas を再描画
        self.canvas.draw()

    def set_x_range(self, x_min_val, x_max_val):
        """
        X軸の表示範囲を固定します。
        :param x_min_val: X軸の最小値
        :param x_max_val: X軸の最大値
        """
        self._x_min = x_min_val
        self._x_max = x_max_val
        self.ax.set_xlim(self._x_min, self._x_max)
        self.canvas.draw()  # 範囲変更を反映させるために再描画


class Example(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PySide6 Matplotlib Trend Chart (Fixed X-axis)")
        self.setGeometry(100, 100, 800, 600)

        self.counter = 0

        self.trend = TrendChart()
        self.setCentralWidget(self.trend)

        # _____________________________________________________________________
        # X軸の範囲を市場の時間に合わせて設定する例
        # 仮に午前9時から午後3時30分までとする
        # 現実の時間に基づいてX軸の範囲を設定する場合、timestampを使うのが一般的です。
        dt_today = datetime.date.today()
        # 仮に2025/06/23 9:00:00 JST から 2025/06/23 15:30:00 JST を範囲とする
        # Matplotlibの時刻軸は数値（Julian dateやtimestamp）で扱うのが普通です
        # time.time() は秒単位のtimestampを返すので、それに合わせます

        # 今日を基準とした特定の時刻のタイムスタンプを取得
        start_time_today = datetime.datetime(
            dt_today.year, dt_today.month, dt_today.day, 9, 0, 0
        )
        end_time_today = datetime.datetime(
            dt_today.year, dt_today.month, dt_today.day, 15, 30, 0
        )
        self.ts_start_market = start_time_today.timestamp()
        self.ts_end_market = end_time_today.timestamp()

        self.trend.set_x_range(self.ts_start_market, self.ts_end_market)
        # _____________________________________________________________________

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_data)
        self.timer.start(100)  # 0.1秒 = 100ミリ秒

    def update_data(self):
        # 現在のタイムスタンプをXデータとして使用
        new_x = time.time()
        new_y = random.randint(0, 100)

        # X軸の範囲外のデータは追加しない、あるいは追加してもプロットしない
        # この例では、範囲外でも追加はしますが、X軸の表示範囲は固定なので見えません。
        # 実際に株価データを扱う場合は、指定時刻外のデータは取得しないか、
        # プロットリストに追加しないロジックが必要になるでしょう。

        # 仮に、市場終了時刻を超えたらタイマーを停止する
        if new_x > self.ts_end_market + 60:  # 終了時刻の1分後で停止
            self.timer.stop()
            print("Market simulation ended.")
            return

        self.trend.appendData(new_x, new_y)


if __name__ == '__main__':
    plt.style.use('dark_background')

    app = QApplication(sys.argv)
    win = Example()
    win.show()
    sys.exit(app.exec())