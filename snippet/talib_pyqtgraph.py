import sys

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMainWindow, QToolBar, QToolButton, QStyle
from talib import stream


class SampleChart(pg.PlotWidget):
    def __init__(self):
        super().__init__()
        self.list_x: list[float] = []
        self.list_y: list[float] = []
        self.list_ma: list[float] = []

        self.line: pg.PlotDataItem = self.plot([], [], pen=pg.mkPen(width=0.5))
        self.ma: pg.PlotDataItem = self.plot([], [], pen=pg.mkPen((255, 128, 0, 192), width=1))

    def setLine(self, x, y):
        self.list_x.append(x)
        self.list_y.append(y)
        self.list_ma.append(stream.SMA(np.array(self.list_y, dtype=float), timeperiod=30))

        self.line.setData(tuple(self.list_x), tuple(self.list_y))
        self.ma.setData(tuple(self.list_x), tuple(self.list_ma))


class SampleTaLib(QMainWindow):
    def __init__(self):
        super().__init__()
        self.file_csv = "sample_data.csv"
        self.df: pd.DataFrame | None = None

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

        self.chart = chart = SampleChart()
        self.setCentralWidget(chart)

        self.timer = timer = QTimer()
        timer.setInterval(100)
        timer.timeout.connect(self.set_new_data)
        self.row: int = 0

    def on_play_clicked(self):
        self.df = pd.read_csv(self.file_csv)
        self.timer.start()
        print("タイマーを動かしました。")

    def on_stop_clicked(self):
        self.timer.stop()
        print("タイマーを止めました。")

    def set_new_data(self):
        x, y = self.df.iloc[self.row]
        self.row += 1
        if len(self.df) <= self.row:
            self.on_stop_clicked()
        else:
            self.chart.setLine(x, y)


def main():
    app = QApplication(sys.argv)
    win = SampleTaLib()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
