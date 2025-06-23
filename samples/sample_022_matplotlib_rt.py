import sys
import random
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class TrendChart(FigureCanvas):
    def __init__(self):
        self.figure = Figure()
        super().__init__(self.figure)

        self.ax = self.figure.add_subplot(111)
        self.ax.grid(True)
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Value")
        self.ax.set_title("Real-time Trend Chart")

        self.list_x = list()
        self.list_y = list()

        # プロットラインの初期化
        self.line, = self.ax.plot(
            self.list_x, self.list_y, label='Trend'
        )

        # 最新の点を表示するためのプロットを初期化
        # marker='o' でサークル、markersize でサイズ、color='red' などで色を指定
        # 初期状態ではデータがないため空のリストでプロット
        self.latest_point, = self.ax.plot(
            [], [],
            marker='o', markersize=8, color='red',
            label='Latest Data'
        )

        # 凡例を表示 (オプション)
        # self.ax.legend()

    def appendData(self, x, y):
        # データを追加
        self.list_x.append(x)
        self.list_y.append(y)

        # グラフを更新
        self.line.set_xdata(self.list_x)
        self.line.set_ydata(self.list_y)

        # 最新の点を更新
        # x_dataとy_dataの最後の要素を取得してプロット
        if self.list_x and self.list_y:  # データが空でないことを確認
            self.latest_point.set_xdata([self.list_x[-1]])
            self.latest_point.set_ydata([self.list_y[-1]])
        else:  # データがない場合は点を非表示にする（初期状態など）
            self.latest_point.set_xdata([])
            self.latest_point.set_ydata([])

        # x軸の範囲を自動調整 (必要に応じて)
        self.ax.relim()
        self.ax.autoscale_view()

        # Canvas を再描画
        self.draw()

    def set_x_range(self, x1, x2):
        self.ax.set_xlim(x1, x2)


class Example(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PySide6 Matplotlib Trend Chart")

        # データカウンタの初期化と最大値
        self.counter = 0
        self.counter_max = 100

        # TrendChart のインスタンス
        self.trend = TrendChart()
        self.trend.set_x_range(0, self.counter_max)
        self.setCentralWidget(self.trend)

        # QTimer の設定 (1秒ごとに update_data メソッドを呼び出す)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_data)
        self.timer.start(100)  # 100ミリ秒 = 0.1秒

    def update_data(self):
        # 新しいデータを生成 (ここではランダムな値を追加)
        self.counter += 1
        new_x = self.counter
        new_y = random.randint(0, 100)

        self.trend.appendData(new_x, new_y)
        if self.counter_max < self.counter:
            self.timer.stop()


if __name__ == '__main__':
    plt.style.use('dark_background')
    app = QApplication(sys.argv)
    win = Example()
    win.show()
    sys.exit(app.exec())
