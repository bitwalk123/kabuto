import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PySide6.QtGui import QFont
import pyqtgraph as pg
import numpy as np

class TrendChartWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQtGraph Trend Charts (X-axis Linked, Height Ratio 3:1, Custom Tick Font, Chart1 X-axis All Hidden)")
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
        # **** X軸の軸ラベルを非表示にするには、空文字列を設定します ****
        self.plot_widget1.setLabel('bottom', '') # <-- ここを修正
        self.plot_widget1.setTitle('Upper Trend')

        # Generate some sample list_data for Chart 1
        x1 = np.linspace(0, 10, 100)
        y1 = np.sin(x1 * 2) + np.random.rand(100) * 10
        self.plot_widget1.plot(x1, y1, pen='b')

        # Set fixed x-axis range for Chart 1
        self.plot_widget1.setXRange(0, 10)

        # **** Chart 1 の X軸ティックラベルを非表示にする (数値部分) ****
        xaxis1 = self.plot_widget1.getAxis('bottom')
        xaxis1.setStyle(showValues=False) # <-- これはティックの数値を非表示にします

        # フォントサイズ設定 (非表示なので効果はないが、もし表示に戻す場合に備えて残す)
        font1 = QFont()
        font1.setPointSize(12)
        xaxis1.setTickFont(font1)

        # Y軸のフォント設定はそのまま
        yaxis1 = self.plot_widget1.getAxis('left')
        font1_y = QFont()
        font1_y.setPointSize(10)
        yaxis1.setTickFont(font1_y)


        # --------------------------------------------------
        # Chart 2: Lower Trend Chart
        # --------------------------------------------------
        self.plot_widget2 = pg.PlotWidget()
        layout.addWidget(self.plot_widget2)

        self.plot_widget2.setLabel('left', 'Value 2')
        self.plot_widget2.setLabel('bottom', 'Time') # Chart 2の軸ラベルは表示したまま
        self.plot_widget2.setTitle('Lower Trend')

        # Generate some sample list_data for Chart 2
        x2 = np.linspace(0, 10, 100)
        y2 = np.cos(x2 * 3) * 100 + np.random.rand(100) * 1000
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