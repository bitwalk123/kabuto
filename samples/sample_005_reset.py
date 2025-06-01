import sys
import datetime
import random
import time

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton
)
from PySide6.QtCore import QTimer, Qt # Qt もインポート
from PySide6.QtGui import QFont

import pyqtgraph as pg
from pyqtgraph import DateAxisItem


class TrendGraph(pg.PlotWidget):
    """
    リアルタイムトレンドグラフ用のPyQtGraphウィジェット。
    時間軸(DateAxisItem)をボトムに持つ。
    """
    def __init__(self):
        super().__init__(
            axisItems={
                'bottom': DateAxisItem(orientation='bottom'),
                'left': pg.AxisItem(orientation='left')
            },
            enableMenu=False
        )
        self.showGrid(x=True, y=True, alpha=0.5)
        pg.setConfigOption('background', 'k')
        pg.setConfigOption('foreground', 'w')

        # --- X軸のティックラベルのフォントを設定 ---
        axis_x_item = self.getAxis('bottom')
        font_x = QFont("monospace")
        font_x.setPointSize(10)
        axis_x_item.tickFont = font_x
        axis_x_item.setStyle(tickTextOffset=8)

        # --- Y軸のティックラベルのフォントを設定 ---
        axis_y_item = self.getAxis('left')
        font_y = QFont("monospace")
        font_y.setPointSize(10)
        axis_y_item.tickFont = font_y
        axis_y_item.setStyle(tickTextOffset=8)


class Example(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQtGraph + PySide6 リアルタイム風トレンドグラフ (リセット機能付き)")
        self.setFixedSize(800, 650) # ボタンの分、少し高さを増やす

        # メインとなるウィジェットとレイアウト
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # グラフウィジェット
        self.chart = TrendGraph()
        main_layout.addWidget(self.chart)

        # リセットボタンの追加
        self.reset_button = QPushButton("リセット")
        self.reset_button.clicked.connect(self.reset_chart) # クリックでリセット処理を呼び出す
        main_layout.addWidget(self.reset_button, alignment=Qt.AlignmentFlag.AlignCenter)


        self.sar_points = pg.ScatterPlotItem(
            size=8,
            pen=pg.mkPen(color=(0, 255, 0), width=1),
            brush=None,
            symbol='o',
            pxMode=True,
            antialias=False
        )
        self.chart.addItem(self.sar_points)

        # データとタイマーの初期化はreset_chartメソッドで行う
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_chart)

        self.reset_chart() # 初回起動時にチャートを初期化

    def reset_chart(self):
        """
        チャートのデータとタイマーをリセットし、最初からプロットを開始する。
        """
        self.timer.stop() # 現在実行中のタイマーがあれば停止

        # データを初期化
        self.x_data = []
        self.y_data = []
        self.sar_points.setData([], []) # ScatterPlotItemのデータをクリア

        # 時間軸の範囲を再設定
        self.start_time = time.time()
        self.end_time = self.start_time + 60
        self.chart.setXRange(self.start_time, self.end_time)
        self.chart.setYRange(0, 100) # Y軸の範囲も念のため再設定

        print("\n--- チャートをリセットし、プロットを再開します ---")
        self.timer.start(1000) # タイマーを再開

    def update_chart(self):
        """
        タイマーイベントで呼び出され、チャートに新しいデータを追加・更新する。
        """
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

        dt_object = datetime.datetime.fromtimestamp(x_val)
        formatted_time = dt_object.strftime("%H:%M:%S")
        # print(f"追加データ: 時刻={formatted_time}, Y={y_val}") # デバッグ出力はコメントアウト


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Example()
    window.show()
    sys.exit(app.exec())