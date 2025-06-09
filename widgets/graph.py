import pyqtgraph as pg
from PySide6.QtGui import QFont
from pyqtgraph import DateAxisItem


class TrendGraph(pg.PlotWidget):
    def __init__(self):
        super().__init__(
            axisItems={
                'bottom': DateAxisItem(orientation='bottom'),
                'left': pg.AxisItem(orientation='left')
            },
            enableMenu=False
        )
        self.setFixedSize(1000, 250)
        self.showGrid(x=True, y=True, alpha=0.5)

        # ★★★ X軸のティックラベルのフォントを設定 ★★★
        axis_x_item = self.getAxis('bottom')
        font_x = QFont("monospace")
        # font_x.setPointSize(10)
        axis_x_item.tickFont = font_x  # setLabelFontではなくtickFont属性に直接代入
        axis_x_item.setStyle(tickTextOffset=8)  # フォント変更後、必要に応じてオフセットを調整

        # ★★★ Y軸のティックラベルのフォントを設定 ★★★
        axis_y_item = self.getAxis('left')
        font_y = QFont("monospace")
        # font_y.setPointSize(10)
        axis_y_item.tickFont = font_y  # setLabelFontではなくtickFont属性に直接代入
        axis_y_item.setStyle(tickTextOffset=8)  # フォント変更後、必要に応じてオフセットを調整
