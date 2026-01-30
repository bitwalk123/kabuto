import logging

import pyqtgraph as pg
import pyqtgraph.exporters
from PySide6.QtCore import QMargins
from pyqtgraph import DateAxisItem

from funcs.plot import trend_label_html
from funcs.setting import get_trend_footer
from structs.res import AppRes
from widgets.misc import TickFont


class CustomYAxisItem(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        return [f"{value:6,.1f}" for value in values]


class TrendGraph(pg.PlotWidget):
    COLOR_MA_1 = "#0f0"
    COLOR_MA_2 = "#faf"
    COLOR_CROSS = "#0ef"
    COLOR_LAST_DOT = "#0f0"
    SIZE_LAST_DOT = 4

    def __init__(self, res: AppRes, dict_ts: dict, dict_setting: dict):
        self.logger = logging.getLogger(__name__)
        self.dict_ts = dict_ts
        self.dict_setting = dict_setting

        # ---------------------------------------------------------------------
        # 軸の設定
        axis_bottom = DateAxisItem(orientation='bottom')
        axis_left = CustomYAxisItem(orientation='left')
        super().__init__(
            axisItems={'bottom': axis_bottom, 'left': axis_left},
            enableMenu=False
        )
        # ---------------------------------------------------------------------
        # ウィンドウのサイズ制約（高さのみ）
        self.setFixedHeight(res.trend_height)
        self.setContentsMargins(QMargins(0, 0, 0, 0))
        # ---------------------------------------------------------------------
        # プロットアイテム
        self.plot_item = self.getPlotItem()
        self.config_plot_item()
        # ---------------------------------------------------------------------
        # 折れ線
        # 株価
        self.line: pg.PlotDataItem = self.plot([], [], pen=pg.mkPen(width=0.25))
        # 移動平均線 1
        self.ma_1: pg.PlotDataItem = self.plot([], [], pen=pg.mkPen(self.COLOR_MA_1, width=1))
        # 移動平均線 2
        self.ma_2: pg.PlotDataItem = self.plot([], [], pen=pg.mkPen(self.COLOR_MA_2, width=1))
        # ---------------------------------------------------------------------
        # 散布図の点
        # 最新値を示すドット
        self.last_dot = pg.ScatterPlotItem(
            size=self.SIZE_LAST_DOT,
            brush=pg.mkBrush(self.COLOR_LAST_DOT),
            pen=None
        )
        self.addItem(self.last_dot)

        self.vline = pg.InfiniteLine(angle=90, pen=pg.mkPen(self.COLOR_CROSS, width=0.75))
        self.addItem(self.vline)

    def config_plot_item(self):
        """
        プロットアイテムの設定
        :return:
        """
        # ---------------------------------------------------------------------
        # x軸範囲（ザラ場時間に固定）
        self.plot_item.setXRange(self.dict_ts["start"], self.dict_ts["end"])
        # ---------------------------------------------------------------------
        # x軸ラベルをフッターとして扱う（日付と設定パラメータ）
        footer = get_trend_footer(self.dict_ts, self.dict_setting)
        self.plot_item.setLabel(axis="bottom", text=trend_label_html(footer, size=7))
        # x軸の余白を設定
        self.plot_item.getAxis('bottom').setHeight(25)
        # ---------------------------------------------------------------------
        # 軸のフォント設定
        font_tick = TickFont()
        self.plot_item.getAxis('bottom').setStyle(tickFont=font_tick)
        self.plot_item.getAxis('left').setStyle(tickFont=font_tick)
        # ---------------------------------------------------------------------
        # グリッド
        self.plot_item.showGrid(x=True, y=True, alpha=0.5)
        # ---------------------------------------------------------------------
        # マウス操作無効化
        self.plot_item.setMouseEnabled(x=False, y=False)
        self.plot_item.hideButtons()
        self.plot_item.setMenuEnabled(False)
        # ---------------------------------------------------------------------
        # 高速化オプション
        self.plot_item.setClipToView(True)

    def setCross(self, x):
        self.vline.setPos(x)

    def setLine(self, line_x, line_y):
        self.line.setData(line_x, line_y)
        # 最新値
        if len(line_x) > 0:
            self.last_dot.setData([line_x[-1]], [line_y[-1]])

    def setTechnicals(self, line_ts, line_ma_1, line_ma_2):
        self.ma_1.setData(line_ts, line_ma_1)
        self.ma_2.setData(line_ts, line_ma_2)

    def setTrendTitle(self, title: str):
        self.setTitle(trend_label_html(title, size=9))

    def save(self, path_img: str):
        """
        チャートをイメージに保存
        https://pyqtgraph.readthedocs.io/en/latest/user_guide/exporting.html
        :param path_img:
        :return:
        """
        exporter = pg.exporters.ImageExporter(self.plot_item)
        exporter.export(path_img)
        self.logger.info(f"{__name__}: チャートが {path_img} に保存されました。")
