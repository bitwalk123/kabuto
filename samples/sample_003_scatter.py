import sys
import datetime
import random
import time

from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PySide6.QtCore import QTimer
import pyqtgraph as pg
from pyqtgraph import DateAxisItem # PyQtGraphのDateAxisItemをインポート


class TrendGraph(pg.PlotWidget):
    """
    リアルタイムトレンドグラフ用のPyQtGraphウィジェット。
    時間軸(DateAxisItem)をボトムに持つ。
    """
    def __init__(self):
        # x軸をDateAxisItemに設定
        super().__init__(
            axisItems={'bottom': DateAxisItem(orientation='bottom')}
        )
        # グリッド線を表示
        self.showGrid(x=True, y=True, alpha=0.5)
        # 背景色をデフォルトで黒に設定（PyQtGraphのsetConfigOptionで全体設定も可能）
        pg.setConfigOption('background', 'k')
        pg.setConfigOption('foreground', 'w')


class Example(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQtGraph + PySide6 リアルタイム風トレンドグラフ (Scatter Plot)")
        self.setFixedSize(800, 600)

        # QMainWindowの中央ウィジェットとしてTrendGraphを設定
        self.chart = TrendGraph()
        self.setCentralWidget(self.chart)

        # X軸の表示範囲を設定 (例: 現在時刻から60秒間)
        self.start_time = time.time()
        self.end_time = self.start_time + 60
        self.chart.setXRange(self.start_time, self.end_time)
        self.chart.setYRange(0, 100) # Y軸の範囲も設定

        # Parabolic SARのデータ点を想定したScatterPlotItemを追加
        # size: 点の大きさ (px)
        # pen: 点の境界線のペン
        # brush: 点の塗りつぶし色
        # symbol: 点の形 ('o' for circle, 's' for square, 't' for triangle, etc.)
        self.sar_points = pg.ScatterPlotItem(
            size=5, # 例として少し小さめに
            pen=pg.mkPen(color=(255, 255, 0), width=1), # 緑色の境界線
            brush=None,
            symbol='o', # 丸い点
            pxMode=True, # サイズをピクセル単位で固定
            antialias=False # アンチエイリアスをオフにすると少し速くなる可能性も
        )
        self.chart.addItem(self.sar_points)

        # プロットするデータを保持するリスト
        self.x_data = [] # 時間 (UNIXタイムスタンプ)
        self.y_data = [] # Y軸の値 (価格など)

        # リアルタイム更新のためのQTimer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_chart)
        self.timer.start(1000) # 1秒ごとに更新

    def update_chart(self):
        """
        タイマーイベントで呼び出され、チャートに新しいデータを追加・更新する。
        """
        current_time = time.time() # 現在のUNIXタイムスタンプ

        # 設定した終了時刻に達したらタイマーを停止
        if current_time > self.end_time:
            self.timer.stop()
            print("リアルタイム更新が終了しました。")
            return

        x_val = current_time
        y_val = random.randint(0, 100) # 0から100のランダムなY値

        # データをリストに追加
        self.x_data.append(x_val)
        self.y_data.append(y_val)

        # Scatter plot のデータを更新
        # setDataは既存のデータを新しいデータで上書きします。
        # リアルタイムでデータを追加していく場合、x_dataとy_dataを常に最新の状態に保ち、
        # その全体をsetDataで渡すのが一般的です。
        self.sar_points.setData(self.x_data, self.y_data)

        # 例として、最新のデータ点をコンソールに出力
        dt_object = datetime.datetime.fromtimestamp(x_val)
        formatted_time = dt_object.strftime("%H:%M:%S")
        print(f"追加データ: 時刻={formatted_time}, Y={y_val}")


if __name__ == "__main__":
    # PyQtGraphのグローバル設定
    # setConfigOptionは、グラフ全体のデフォルト設定を変更します。
    # 'background' は背景色、'foreground' は前景（軸のラベル、ティックなど）の色
    pg.setConfigOption('background', 'k') # 黒背景 (ダークモード風)
    pg.setConfigOption('foreground', 'w') # 白前景 (テキストなど)

    app = QApplication(sys.argv)
    window = Example()
    window.show()
    sys.exit(app.exec())