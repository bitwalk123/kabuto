import sys
from typing import Optional

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMainWindow, QStyle, QToolBar, QToolButton
from talib import stream


class SampleChart(pg.PlotWidget):
    def __init__(self, ma_period: int = 30) -> None:
        super().__init__()
        self.ma_period = ma_period

        # リストで保持（append が高速）
        self.data_x: list[float] = []
        self.data_y: list[float] = []
        self.data_ma: list[float] = []

        # streaming APIの状態を保持するためのバッファ
        self.y_buffer = np.array([], dtype=float)

        self.line: pg.PlotDataItem = self.plot([], [], pen=pg.mkPen(width=0.5))
        self.ma: pg.PlotDataItem = self.plot([], [], pen=pg.mkPen((0, 255, 0, 192), width=1))

    def add_point(self, x: float, y: float) -> None:
        """新しいデータポイントを追加"""
        self.data_x.append(x)
        self.data_y.append(y)

        # バッファを更新（直近ma_period個のデータのみ保持）
        self.y_buffer = np.append(self.y_buffer, y)
        if len(self.y_buffer) > self.ma_period:
            self.y_buffer = self.y_buffer[-self.ma_period:]

        # streaming APIで移動平均を計算
        if len(self.y_buffer) >= self.ma_period:
            self.data_ma.append(stream.SMA(self.y_buffer, timeperiod=self.ma_period))

        # グラフを更新
        self.line.setData(self.data_x, self.data_y)  # type: ignore

        # MA期間に達したらMAラインを表示
        if len(self.data_ma) > 0:
            ma_start = self.ma_period - 1
            self.ma.setData(self.data_x[ma_start:], self.data_ma)  # type: ignore


class SampleTaLib(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.file_csv = "sample_data.zip"  # ZIP圧縮されたCSVファイルを読み込む
        self.df: Optional[pd.DataFrame] = None
        self.row: int = 0

        self.setWindowTitle("TA-Lib Streaming API Demo")
        self.resize(800, 600)

        # ツールバーの設定
        toolbar = QToolBar()

        but_play = QToolButton()
        but_play.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        but_play.clicked.connect(self.on_play_clicked)
        toolbar.addWidget(but_play)

        but_stop = QToolButton()
        but_stop.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        but_stop.clicked.connect(self.on_stop_clicked)
        toolbar.addWidget(but_stop)

        self.addToolBar(toolbar)

        # チャートの設定
        self.chart = SampleChart(ma_period=30)
        self.setCentralWidget(self.chart)

        # タイマーの設定
        self.timer = QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.set_new_data)

    def on_play_clicked(self) -> None:
        """再生ボタンクリック時の処理"""
        try:
            # データ未読み込みの場合のみ読み込む
            if self.df is None:
                self.df = pd.read_csv(self.file_csv)
                print(f"CSVファイルを読み込みました: {len(self.df)}行")

                # データの検証
                if len(self.df.columns) < 2:
                    print("エラー: CSVファイルには少なくとも2列必要です")
                    return

            if not self.timer.isActive():
                self.timer.start()
                print("タイマーを開始しました。")

        except FileNotFoundError:
            print(f"エラー: ファイル '{self.file_csv}' が見つかりません")
        except Exception as e:
            print(f"エラー: {e}")

    def on_stop_clicked(self) -> None:
        """停止ボタンクリック時の処理"""
        if self.timer.isActive():
            self.timer.stop()
            print("タイマーを停止しました。")

    def set_new_data(self) -> None:
        """新しいデータをチャートに追加"""
        if self.df is None or self.row >= len(self.df):
            self.on_stop_clicked()
            print("データの最後に到達しました。")
            return

        try:
            x, y = self.df.iloc[self.row, 0], self.df.iloc[self.row, 1]
            self.chart.add_point(float(x), float(y))
            self.row += 1
        except Exception as e:
            print(f"データ追加エラー: {e}")
            self.on_stop_clicked()


def main() -> None:
    app = QApplication(sys.argv)
    win = SampleTaLib()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
