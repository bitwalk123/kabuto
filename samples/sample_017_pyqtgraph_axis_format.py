import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PySide6.QtGui import QFont, QFontDatabase  # QFontDatabaseもインポート
import pyqtgraph as pg
import numpy as np
import datetime


# カスタムY軸アイテムクラス
class CustomYAxisItem(pg.AxisItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def tickStrings(self, values, scale, spacing):
        strings = []
        for value in values:
            formatted_value = f"{value:5.0f}"
            strings.append(formatted_value)
        return strings


# DateAxisItemのダミー定義
class DateAxisItem(pg.AxisItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def tickStrings(self, values, scale, spacing):
        strings = []
        for v in values:
            try:
                dt_object = datetime.datetime.fromtimestamp(v)
                strings.append(dt_object.strftime('%H:%M:%S'))
            except (ValueError, OSError):
                strings.append('')
        return strings


class TrendChartWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQtGraph Trend Charts (Custom Y-Axis Format, Generic Monospace Font)")
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # **** 汎用的な等幅フォントファミリーの指定 (修正箇所) ****
        # まずは通常のQFontオブジェクトを作成
        generic_monospace_font = QFont()
        # setStyleHintを使って汎用フォントファミリーを指定
        generic_monospace_font.setStyleHint(QFont.StyleHint.Monospace)
        generic_monospace_font.setPointSize(12)  # サイズ設定

        generic_monospace_font_small = QFont()
        generic_monospace_font_small.setStyleHint(QFont.StyleHint.Monospace)
        generic_monospace_font_small.setPointSize(10)  # サイズ設定

        # デバッグ用：実際にどのフォントが選ばれたかを確認 (オプション)
        # print(f"Selected Monospace Font: {generic_monospace_font.family()}")

        # --------------------------------------------------
        # Chart 1: Upper Trend Chart
        # --------------------------------------------------
        self.plot_widget1 = pg.PlotWidget(
            axisItems={
                'bottom': DateAxisItem(orientation='bottom'),
                'left': CustomYAxisItem(orientation='left')
            },
            enableMenu=False
        )
        layout.addWidget(self.plot_widget1)

        self.plot_widget1.setMouseEnabled(x=False, y=False)

        self.plot_widget1.setLabel('left', 'Value 1')
        self.plot_widget1.setLabel('bottom', '')
        self.plot_widget1.setTitle('Upper Trend')

        now = datetime.datetime.now().timestamp()
        x1 = np.linspace(now - 10, now, 100)
        y1 = (np.sin((x1 - now) * 2) + np.random.rand(100) * 0.5) * 12345.67
        self.plot_widget1.plot(x1, y1, pen='b')
        self.plot_widget1.setXRange(x1[0], x1[-1])

        xaxis1 = self.plot_widget1.getAxis('bottom')
        xaxis1.setStyle(showValues=False)

        # QFont.Monospace をX軸ティックラベルに適用
        xaxis1.setTickFont(generic_monospace_font)

        # QFont.Monospace をY軸ティックラベルに適用
        self.plot_widget1.getAxis('left').setTickFont(generic_monospace_font_small)

        # --------------------------------------------------
        # Chart 2: Lower Trend Chart
        # --------------------------------------------------
        self.plot_widget2 = pg.PlotWidget(
            axisItems={
                'bottom': DateAxisItem(orientation='bottom'),
                'left': CustomYAxisItem(orientation='left')
            },
            enableMenu=False
        )
        layout.addWidget(self.plot_widget2)

        self.plot_widget2.setMouseEnabled(x=False, y=False)

        self.plot_widget2.setLabel('left', 'Value 2')
        self.plot_widget2.setLabel('bottom', 'Time')
        self.plot_widget2.setTitle('Lower Trend')

        x2 = np.linspace(now - 10, now, 100)
        y2 = (np.cos((x2 - now) * 3) * 100 + np.random.rand(100) * 30) * 0.123
        self.plot_widget2.plot(x2, y2, pen='r')
        self.plot_widget2.setXRange(x2[0], x2[-1])

        # QFont.Monospace をX軸ティックラベルに適用
        self.plot_widget2.getAxis('bottom').setTickFont(generic_monospace_font)

        # QFont.Monospace をY軸ティックラベルに適用
        self.plot_widget2.getAxis('left').setTickFont(generic_monospace_font_small)

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