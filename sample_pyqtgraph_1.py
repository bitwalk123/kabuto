import sys
import random
import pyqtgraph as pg
from PySide6.QtWidgets import QApplication, QMainWindow


class TrendGraph(pg.PlotWidget):
    def __init__(self, x_list):
        super().__init__()

        self.x_list = x_list
        self.y_list = []
        self.index = 0

        # グリッド
        self.showGrid(x=True, y=True, alpha=0.5)

        # 折れ線（最初は空）
        self.curve = self.plot([], [], pen=pg.mkPen('c', width=2))

        # 最新点（X マーク）
        self.latest = pg.ScatterPlotItem(
            size=10,
            pen=pg.mkPen('r', width=2),
            brush=None,
            symbol='x'
        )
        self.addItem(self.latest)

        # タイマー
        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update_point)
        self.timer.start(100)

    def update_point(self):
        if self.index >= len(self.x_list):
            return

        # y をランダムに生成
        y = random.uniform(-1.0, 1.0)
        self.y_list.append(y)

        x = self.x_list[self.index]
        self.index += 1

        # 折れ線更新
        self.curve.setData(self.x_list[:self.index], self.y_list)

        # 最新点更新
        self.latest.setData([x], [y])


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # x は固定（例として 0〜99）
        x = list(range(100))

        self.graph = TrendGraph(x)
        self.setCentralWidget(self.graph)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    app.exec()
