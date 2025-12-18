import pyqtgraph as pg
from PySide6.QtCore import QMargins
from PySide6.QtGui import QFont
from pyqtgraph import DateAxisItem

from structs.res import AppRes


class CustomYAxisItem(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        return [f"{value:6,.0f}" for value in values]


class TrendGraph(pg.PlotWidget):
    def __init__(self, res: AppRes):
        axis_bottom = DateAxisItem(orientation='bottom')
        axis_left = CustomYAxisItem(orientation='left')
        super().__init__(
            axisItems={'bottom': axis_bottom, 'left': axis_left},
            enableMenu=False
        )
        # ウィンドウのサイズ制約（高さのみ）
        self.setFixedHeight(res.trend_height)
        self.setContentsMargins(QMargins(0, 0, 0, 0))

        # マウス操作無効化
        self.setMouseEnabled(x=False, y=False)
        self.hideButtons()
        self.setMenuEnabled(False)

        # フォント設定
        font_small = QFont()
        font_small.setStyleHint(QFont.StyleHint.Monospace)
        font_small.setPointSize(9)
        self.getAxis('bottom').setStyle(tickFont=font_small)
        self.getAxis('left').setStyle(tickFont=font_small)

        # プロットアイテム
        self.plot_item = plot_item = self.getPlotItem()

        # グリッド
        plot_item.showGrid(x=True, y=True, alpha=0.5)
        # 高速化オプション
        plot_item.setClipToView(True)

        # 折れ線
        self.line: pg.PlotDataItem = self.plot([], [], pen=pg.mkPen(width=0.5))
        # 移動平均線 1
        self.ma_1: pg.PlotDataItem = self.plot([], [], pen=pg.mkPen("#0f0", width=0.75))
        # 移動平均線 2
        self.ma_2: pg.PlotDataItem = self.plot([], [], pen=pg.mkPen("#faf", width=1))

    def setLine(self, line_x, line_y):
        self.line.setData(line_x, line_y)

    def setTechnicals(self, line_ts, line_ma_1, line_ma_2):
        self.ma_1.setData(line_ts, line_ma_1)
        self.ma_2.setData(line_ts, line_ma_2)

    def setTrendTitle(self, title: str):
        html = f"<span style='font-size: 9pt; font-family: monospace;'>{title}</span>"
        self.setTitle(html)
