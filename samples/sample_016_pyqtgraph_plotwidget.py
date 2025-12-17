import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PySide6.QtGui import QFont
import pyqtgraph as pg
import numpy as np

class TrendChartWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQtGraph Trend Charts (X-axis Linked, Height Ratio 3:1, Interaction Disabled)")
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # --------------------------------------------------
        # Chart 1: Upper Trend Chart
        # --------------------------------------------------
        self.plot_widget1 = pg.PlotWidget()
        layout.addWidget(self.plot_widget1)

        self.plot_widget1.setLabel('left', 'Value 1')
        self.plot_widget1.setLabel('bottom', '') # X軸の軸ラベルを非表示
        self.plot_widget1.setTitle('Upper Trend')

        # Generate some sample list_item for Chart 1
        x1 = np.linspace(0, 10, 100)
        y1 = np.sin(x1 * 2) + np.random.rand(100) * 0.5
        self.plot_widget1.plot(x1, y1, pen='b')

        # Set fixed x-axis range for Chart 1
        self.plot_widget1.setXRange(0, 10)

        # Chart 1 の X軸ティックラベルを非表示にする
        xaxis1 = self.plot_widget1.getAxis('bottom')
        xaxis1.setStyle(showValues=False)

        # フォントサイズ設定
        font1 = QFont()
        font1.setPointSize(12)
        xaxis1.setTickFont(font1)

        yaxis1 = self.plot_widget1.getAxis('left')
        font1_y = QFont()
        font1_y.setPointSize(10)
        yaxis1.setTickFont(font1_y)

        # **** ここが修正点 ****
        # マウスによるパン/ズーム (ドラッグとホイール) をX, Y方向ともに無効化
        self.plot_widget1.setMouseEnabled(x=False, y=False)
        # 右クリックメニューを無効化
        self.plot_widget1.setMenuEnabled(False)


        # --------------------------------------------------
        # Chart 2: Lower Trend Chart
        # --------------------------------------------------
        self.plot_widget2 = pg.PlotWidget()
        layout.addWidget(self.plot_widget2)

        self.plot_widget2.setLabel('left', 'Value 2')
        self.plot_widget2.setLabel('bottom', 'Time')
        self.plot_widget2.setTitle('Lower Trend')

        # Generate some sample list_item for Chart 2
        x2 = np.linspace(0, 10, 100)
        y2 = np.cos(x2 * 3) * 100 + np.random.rand(100) * 30
        self.plot_widget2.plot(x2, y2, pen='r')

        # Set fixed x-axis range for Chart 2
        self.plot_widget2.setXRange(0, 10)

        # Chart 2 のティックラベルのフォントサイズ設定
        xaxis2 = self.plot_widget2.getAxis('bottom')
        font2 = QFont()
        font2.setPointSize(12)
        xaxis2.setTickFont(font2)

        yaxis2 = self.plot_widget2.getAxis('left')
        font2_y = QFont()
        font2_y.setPointSize(10)
        yaxis2.setTickFont(font2_y)

        # **** ここも修正点 ****
        self.plot_widget2.setMouseEnabled(x=False, y=False)
        self.plot_widget2.setMenuEnabled(False)


        # --------------------------------------------------
        # Link x-axes
        # --------------------------------------------------
        self.plot_widget2.setXLink(self.plot_widget1)

        # --------------------------------------------------
        # Set stretch factor for height ratio
        # --------------------------------------------------
        layout.setStretchFactor(self.plot_widget1, 3)
        layout.setStretchFactor(self.plot_widget2, 1)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TrendChartWindow()
    window.show()
    sys.exit(app.exec())