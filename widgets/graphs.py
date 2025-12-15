import pyqtgraph as pg
from PySide6.QtGui import QFont
from pyqtgraph import DateAxisItem


class CustomYAxisItem(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        return [f"{value:5,.0f}" for value in values]


class TrendGraph(pg.PlotWidget):
    def __init__(self):
        axis_bottom = DateAxisItem(orientation='bottom')
        axis_left = CustomYAxisItem(orientation='left')
        super().__init__(
            axisItems={'bottom': axis_bottom, 'left': axis_left},
            enableMenu=False
        )
        # ウィンドウのサイズ制約
        self.setMinimumWidth(1000)
        self.setFixedHeight(200)

        # マウス操作無効化
        self.setMouseEnabled(x=False, y=False)
        self.hideButtons()
        self.setMenuEnabled(False)

        # フォント設定
        font = QFont()
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPointSize(10)

        self.getAxis('bottom').setStyle(tickFont=font)
        self.getAxis('left').setStyle(tickFont=font)

        plot_item = self.getPlotItem()
        # グリッド
        plot_item.showGrid(x=True, y=True, alpha=0.5)
        # 高速化オプション
        plot_item.setClipToView(True)
