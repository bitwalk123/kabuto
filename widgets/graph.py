import pyqtgraph as pg
from PySide6.QtGui import QFont
from pyqtgraph import DateAxisItem


class CustomYAxisItem(pg.AxisItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def tickStrings(self, values, scale, spacing):
        strings = []
        for value in values:
            formatted_value = f"{value:5.0f}"
            strings.append(formatted_value)
        return strings


class TrendGraph(pg.PlotWidget):
    def __init__(self):
        super().__init__(
            # axisItems={
            #    'bottom': DateAxisItem(orientation='bottom'),
            #    'left': pg.AxisItem(orientation='left')
            # },
            axisItems={
                'bottom': DateAxisItem(orientation='bottom'),
                'left': CustomYAxisItem(orientation='left')
            },
            enableMenu=False
        )
        self.setFixedSize(1000, 200)
        self.showGrid(x=True, y=True, alpha=0.5)

        # **** マウスによるパン/ズーム無効化 ****
        self.setMouseEnabled(x=False, y=False)

        # **** 汎用的な等幅フォントファミリーの指定 (修正箇所) ****
        # まずは通常のQFontオブジェクトを作成
        generic_monospace_font = QFont()
        # setStyleHintを使って汎用フォントファミリーを指定
        generic_monospace_font.setStyleHint(QFont.StyleHint.Monospace)
        generic_monospace_font.setPointSize(12)  # サイズ設定

        generic_monospace_font_small = QFont()
        generic_monospace_font_small.setStyleHint(QFont.StyleHint.Monospace)
        generic_monospace_font_small.setPointSize(10)  # サイズ設定

        # ★★★ X軸のティックラベルのフォントを設定 ★★★
        #axis_x_item = self.getAxis('bottom')
        #font_x = QFont("monospace")

        # font_x.setPointSize(10)
        #axis_x_item.tickFont = font_x  # setLabelFontではなくtickFont属性に直接代入
        self.getAxis('bottom').setTickFont(generic_monospace_font_small)
        #axis_x_item.setStyle(tickTextOffset=8)  # フォント変更後、必要に応じてオフセットを調整

        # ★★★ Y軸のティックラベルのフォントを設定 ★★★
        #axis_y_item = self.getAxis('left')
        #font_y = QFont("monospace")

        # font_y.setPointSize(10)
        #axis_y_item.tickFont = font_y  # setLabelFontではなくtickFont属性に直接代入
        # QFont.Monospace をY軸ティックラベルに適用
        self.getAxis('left').setTickFont(generic_monospace_font_small)
        #axis_y_item.setStyle(tickTextOffset=8)  # フォント変更後、必要に応じてオフセットを調整


class TrendGraph2(TrendGraph):
    def __init__(self):
        super().__init__()
        self.setFixedSize(1000, 100)
