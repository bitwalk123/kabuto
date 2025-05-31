import sys
import datetime
import random
import time

from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PySide6.QtCore import QTimer
from PySide6.QtGui import QFont # QFontをインポート

import pyqtgraph as pg
from pyqtgraph import DateAxisItem


class TrendGraph(pg.PlotWidget):
    """
    リアルタイムトレンドグラフ用のPyQtGraphウィジェット。
    時間軸(DateAxisItem)をボトムに持つ。
    """
    def __init__(self):
        # Y軸を明示的にAxisItemとして追加 (フォント設定のために必要)
        super().__init__(
            axisItems={
                'bottom': DateAxisItem(orientation='bottom'),
                'left': pg.AxisItem(orientation='left')
            }
        )
        self.showGrid(x=True, y=True, alpha=0.5)
        pg.setConfigOption('background', 'k')
        pg.setConfigOption('foreground', 'w')

        # ★★★ X軸のティックラベルのフォントを設定 ★★★
        axis_x_item = self.getAxis('bottom')
        font_x = QFont("monospace")
        #font_x.setPointSize(10)
        axis_x_item.tickFont = font_x # setLabelFontではなくtickFont属性に直接代入
        axis_x_item.setStyle(tickTextOffset=8) # フォント変更後、必要に応じてオフセットを調整

        # ★★★ Y軸のティックラベルのフォントを設定 ★★★
        axis_y_item = self.getAxis('left')
        font_y = QFont("monospace")
        #font_y.setPointSize(10)
        axis_y_item.tickFont = font_y # setLabelFontではなくtickFont属性に直接代入
        axis_y_item.setStyle(tickTextOffset=8) # フォント変更後、必要に応じてオフセットを調整


class Example(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQtGraph + PySide6 リアルタイム風トレンドグラフ (等幅フォント)")
        self.setFixedSize(800, 600)

        self.chart = TrendGraph()
        self.setCentralWidget(self.chart)

        self.start_time = time.time()
        self.end_time = self.start_time + 60
        self.chart.setXRange(self.start_time, self.end_time)
        self.chart.setYRange(0, 100)

        self.sar_points = pg.ScatterPlotItem(
            size=8,
            pen=pg.mkPen(color=(0, 255, 0), width=1),
            brush=None,
            symbol='o',
            pxMode=True,
            antialias=False
        )
        self.chart.addItem(self.sar_points)

        self.x_data = []
        self.y_data = []

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_chart)
        self.timer.start(1000)

    def update_chart(self):
        current_time = time.time()
        if current_time > self.end_time:
            self.timer.stop()
            print("リアルタイム更新が終了しました。")
            return

        x_val = current_time
        y_val = random.randint(0, 100)

        self.x_data.append(x_val)
        self.y_data.append(y_val)

        self.sar_points.setData(self.x_data, self.y_data)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Example()
    window.show()
    sys.exit(app.exec())