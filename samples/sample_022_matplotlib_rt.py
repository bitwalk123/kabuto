import sys
import random
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class TrendChartWidget(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        # データの初期化
        self.counter = 0
        self.counter_max = 100
        self.x_data = []
        self.y_data = []

        # Matplotlib の Figure と Canvas を作成
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.setCentralWidget(self.canvas)

        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlim(0, self.counter_max)
        self.ax.grid()
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Value")
        self.ax.set_title("Real-time Trend Chart")

        # プロットラインの初期化
        self.line, = self.ax.plot(
            self.x_data, self.y_data,
            label='Trend'
        )  # labelを追加

        # 最新の点を表示するためのプロットを初期化
        # marker='o' でサークル、markersize でサイズ、color='red' などで色を指定
        # 初期状態ではデータがないため空のリストでプロット
        self.latest_point, = self.ax.plot(
            [], [],
            marker='o',
            markersize=8,
            color='red',
            label='Latest Data'
        )

        # 凡例を表示 (オプション)
        # self.ax.legend()

        # QTimer の設定 (1秒ごとに update_data メソッドを呼び出す)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_data)
        self.timer.start(100)  # 100ミリ秒 = 0.1秒

    def update_data(self):
        # 新しいデータを生成 (ここではランダムな値を追加)
        self.counter += 1
        new_x = self.counter
        new_y = random.randint(0, 100)

        # データを追加
        self.x_data.append(new_x)
        self.y_data.append(new_y)

        # グラフを更新
        self.line.set_xdata(self.x_data)
        self.line.set_ydata(self.y_data)

        # 最新の点を更新
        # x_dataとy_dataの最後の要素を取得してプロット
        if self.x_data and self.y_data:  # データが空でないことを確認
            self.latest_point.set_xdata([self.x_data[-1]])
            self.latest_point.set_ydata([self.y_data[-1]])
        else:  # データがない場合は点を非表示にする（初期状態など）
            self.latest_point.set_xdata([])
            self.latest_point.set_ydata([])

        # x軸の範囲を自動調整 (必要に応じて)
        self.ax.relim()
        self.ax.autoscale_view()

        # Canvas を再描画
        self.canvas.draw()

        if self.counter_max < self.counter:
            self.timer.stop()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TrendChartWidget()
    window.setWindowTitle("PySide6 Matplotlib Trend Chart")
    #window.setGeometry(100, 100, 800, 600)
    window.show()
    sys.exit(app.exec())
