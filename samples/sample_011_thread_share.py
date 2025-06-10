import math
import sys
import random

import numpy as np
import pyqtgraph as pg
from scipy.interpolate import make_smoothing_spline
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot


class DataGeneratorWorker(QObject):
    notifyNewData = Signal(int, float)
    notifySmoothLine = Signal(np.ndarray, np.ndarray)

    def __init__(self, max_data: int, parent=None):
        super().__init__(parent)
        self.x_data = np.empty(max_data, dtype=np.int64)
        self.y_data = np.empty(max_data, dtype=np.float64)

    @Slot(int)
    def generateNewData(self, counter: int):
        """
        サンプルデータ
        :param counter:
        :return:
        """
        x = counter
        y = 5 * math.sin(x / 10.) + random.random() + 10
        self.notifyNewData.emit(x, y)

        # スムージング処理
        self.x_data[counter] = x
        self.y_data[counter] = y
        if counter > 5:
            spl = make_smoothing_spline(
                self.x_data[0:counter + 1],
                self.y_data[0:counter + 1]
            )
            ys_data = spl(self.x_data[0:counter + 1])
            self.notifySmoothLine.emit(self.x_data[0:counter + 1], ys_data)
            # スムージングしたデータに対して Parabolic SAR の計算を適用したい！


class ThreadDataGenerator(QThread):
    requestNewData = Signal(int)
    threadReady = Signal()

    def __init__(self, max_data: int, parent=None):
        super().__init__(parent)
        self.worker = DataGeneratorWorker(max_data)
        self.worker.moveToThread(self)

        self.started.connect(self.thread_ready)
        self.requestNewData.connect(self.worker.generateNewData)

    def thread_ready(self):
        self.threadReady.emit()

    def run(self):
        self.exec()  # イベントループを開始


class TrendGraph(pg.PlotWidget):
    def __init__(self, max_data: int):
        super().__init__()
        # データを保持するリスト
        self.x_data = []
        self.y_data = []

        self.showGrid(x=True, y=True, alpha=0.5)
        self.setXRange(0, max_data)

        # データ点
        self.data_points = pg.ScatterPlotItem(
            size=5,
            pen=pg.mkPen(color=(0, 255, 255), width=1),
            brush=pg.mkBrush(color=(0, 255, 255)),
            symbol='o',
            pxMode=True,
            antialias=False
        )
        self.addItem(self.data_points)

        self.smoothed_line = pg.PlotDataItem(
            pen=pg.mkPen(color=(255, 255, 0), width=1),
            pxMode=True,
            antialias=False
        )
        self.addItem(self.smoothed_line)

    def addPoints(self, x, y):
        # データをリストに追加
        self.x_data.append(x)
        self.y_data.append(y)

        self.data_points.setData(self.x_data, self.y_data)

        # 例として、最新のデータ点をコンソールに出力
        print(f"追加データ: X={x}, Y={y}")

    def updateSmoothedLine(self, x_array: np.ndarray, y_array: np.ndarray):
        self.smoothed_line.setData(x_array, y_array)


class Example(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQtGraph + PySide6 リアルタイム風トレンドグラフ (Scatter Plot)")
        self.setFixedSize(800, 600)

        # データの最大数とカウンタ
        self.max_data = 180
        self.count = 0

        self.chart = TrendGraph(self.max_data)
        self.setCentralWidget(self.chart)

        self.thread = thread = ThreadDataGenerator(self.max_data)
        thread.threadReady.connect(self.on_thread_ticker_ready)
        thread.worker.notifyNewData.connect(self.chart.addPoints)
        thread.worker.notifySmoothLine.connect(self.chart.updateSmoothedLine)
        thread.start()

        # リアルタイム更新のためのQTimer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_chart)
        self.timer.start(1000)  # 1秒ごとに更新

    def closeEvent(self, event: QCloseEvent):
        if self.thread.isRunning():
            print("Stopping thread...")
            self.thread.quit()
            self.thread.wait()
            print("The thread safely terminated.")

    def on_thread_ticker_ready(self):
        print("Thread is ready!")

    def update_chart(self):
        if self.count >= self.max_data:
            self.timer.stop()
            print("リアルタイム更新が終了しました。")
            return

        self.thread.requestNewData.emit(self.count)
        self.count += 1


if __name__ == "__main__":
    pg.setConfigOption('background', 'k')  # 黒背景 (ダークモード風)
    pg.setConfigOption('foreground', 'w')  # 白前景 (テキストなど)

    app = QApplication(sys.argv)
    window = Example()
    window.show()
    sys.exit(app.exec())
