import pyqtgraph as pg
from PySide6.QtGui import QFont
from pyqtgraph import DateAxisItem

from structs.res import AppRes


class CustomYAxisItem(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        return [f"{value:6,.0f}" for value in values]


class TrendGraph(pg.PlotWidget):
    def __init__(self, res:AppRes):
        axis_bottom = DateAxisItem(orientation='bottom')
        axis_left = CustomYAxisItem(orientation='left')
        super().__init__(
            axisItems={'bottom': axis_bottom, 'left': axis_left},
            enableMenu=False
        )
        # ウィンドウのサイズ制約（高さのみ）
        self.setFixedHeight(res.trend_height)

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
        self.line: pg.PlotDataItem = self.plot([], [], pen=pg.mkPen(width=0.75))

    def setLine(self, x, y):
        self.line.setData(x, y)

    def setTrendTitle(self, title: str):
        html = f"<span style='font-size: 9pt; font-family: monospace;'>{title}</span>"
        self.setTitle(html)
