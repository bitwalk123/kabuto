import sys
import datetime
import random
import time
import csv

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QFileDialog,
    QHBoxLayout  # ★★★ ここを追加/修正 ★★★
)
from PySide6.QtCore import QTimer, Qt
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
        self.setWindowTitle("PyQtGraph + PySide6 リアルタイム風トレンドグラフ (データ保存機能付き)")
        self.setFixedSize(800, 650)  # ボタンの分、少し高さを増やす

        # メインとなるウィジェットとレイアウト
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # グラフウィジェット
        self.chart = TrendGraph()
        main_layout.addWidget(self.chart)

        # ボタン用の水平レイアウトを作成
        # ★★★ ここを修正：QHBoxLayoutをPySide6.QtWidgetsから直接使用 ★★★
        button_layout = QHBoxLayout()

        # リセットボタンの追加
        self.reset_button = QPushButton("リセット")
        self.reset_button.clicked.connect(self.reset_chart)
        button_layout.addWidget(self.reset_button)

        # データ保存ボタンの追加
        self.save_button = QPushButton("データ保存")
        self.save_button.clicked.connect(self.save_data_to_file)
        button_layout.addWidget(self.save_button)

        main_layout.addLayout(button_layout)  # メインレイアウトにボタンレイアウトを追加

        self.sar_points = pg.ScatterPlotItem(
            size=8,
            pen=pg.mkPen(color=(0, 255, 0), width=1),
            brush=None,
            symbol='o',
            pxMode=True,
            antialias=False
        )
        self.chart.addItem(self.sar_points)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_chart)

        self.reset_chart()  # 初回起動時にチャートを初期化

    def reset_chart(self):
        """
        チャートのデータとタイマーをリセットし、最初からプロットを開始する。
        """
        self.timer.stop()

        self.x_data = []
        self.y_data = []
        self.sar_points.setData([], [])

        self.start_time = time.time()
        self.end_time = self.start_time + 60
        self.chart.setXRange(self.start_time, self.end_time)
        self.chart.setYRange(0, 100)

        print("\n--- チャートをリセットし、プロットを再開します ---")
        self.timer.start(1000)

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

    def save_data_to_file(self):
        """
        現在のデータをファイルに保存するダイアログを表示し、CSV形式で保存する。
        """
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "データを保存",
            "",
            "CSV Files (*.csv);;All Files (*)",
            "CSV Files (*.csv)"
        )

        if file_path:
            try:
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Timestamp", "Datetime", "Value"])
                    for i in range(len(self.x_data)):
                        timestamp = self.x_data[i]
                        value = self.y_data[i]
                        dt_object = datetime.datetime.fromtimestamp(timestamp)
                        writer.writerow([timestamp, dt_object.strftime("%Y-%m-%d %H:%M:%S.%f"), value])
                print(f"データが '{file_path}' に保存されました。")
            except Exception as e:
                print(f"データの保存中にエラーが発生しました: {e}")
        else:
            print("データの保存がキャンセルされました。")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Example()
    window.show()
    sys.exit(app.exec())