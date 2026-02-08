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


class TrendChart(pg.PlotWidget):
    COLOR_MA_1 = (0, 255, 0, 192)
    COLOR_VWAP = (255, 0, 192, 192)
    COLOR_GOLDEN = (255, 0, 204, 220)
    COLOR_DEAD = (0, 191, 255, 220)
    COLOR_DISPARITY = (255, 255, 0, 192)
    COLOR_LAST_DOT = (0, 255, 0, 255)
    SIZE_LAST_DOT = 4

    def __init__(self, res: AppRes, dict_ts: dict, dict_setting: dict):
        self.logger = logging.getLogger(__name__)
        self.res = res
        self.dict_ts = dict_ts
        self.dict_setting = dict_setting

        # ---------------------------------------------------------------------
        # 軸の設定
        axis_bottom = DateAxisItem(orientation='bottom')
        axis_left = CustomYAxisItem(orientation='left')
        # フォント設定
        font_mono = TickFont(self.res)
        axis_bottom.setStyle(tickFont=font_mono)
        axis_left.setStyle(tickFont=font_mono)

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
        self.line.setZValue(90)
        # 移動平均線 1
        self.ma_1: pg.PlotDataItem = self.plot([], [], pen=pg.mkPen(self.COLOR_MA_1, width=1))
        self.ma_1.setZValue(60)
        # VWAP
        self.vwap: pg.PlotDataItem = self.plot([], [], pen=pg.mkPen(self.COLOR_VWAP, width=1))
        self.vwap.setZValue(50)
        # 乖離率
        self.disparity: pg.PlotDataItem = self.plot([], [], pen=pg.mkPen(self.COLOR_DISPARITY, width=1))
        self.disparity.setZValue(50)
        # y = 0 のライン
        self.zero_line = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen('#fff', width=0.5))
        self.zero_line.setZValue(0)
        self.addItem(self.zero_line)
        # ---------------------------------------------------------------------
        # 散布図の点
        # 最新値を示すドット
        self.last_dot = pg.ScatterPlotItem(
            size=self.SIZE_LAST_DOT,
            brush=pg.mkBrush(self.COLOR_LAST_DOT),
            pen=None
        )
        self.last_dot.setZValue(100)
        self.addItem(self.last_dot)
        # クロス線（ゴールデン・クロス）
        self.vline_golden = pg.InfiniteLine(
            angle=90,
            pen=pg.mkPen(self.COLOR_GOLDEN, width=0.75)
        )
        self.vline_golden.setZValue(20)
        self.addItem(self.vline_golden)
        # クロス線（デッド・クロス）
        self.vline_dead = pg.InfiniteLine(
            angle=90,
            pen=pg.mkPen(self.COLOR_DEAD, width=0.75)
        )
        self.vline_dead.setZValue(20)
        self.addItem(self.vline_dead)

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
        self.plot_item.getAxis('bottom').setHeight(28)
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

    def setCrossDead(self, x):
        self.vline_dead.setPos(x)

    def setCrossGolden(self, x):
        self.vline_golden.setPos(x)

    def setLine(self, line_x, line_y):
        # トレンド線
        self.line.setData(line_x, line_y)

    def setDot(self, x, y):
        # 最新値
        self.last_dot.setData(x, y)

    def setTechnicals(
            self,
            line_ts,
            line_ma_1,
            line_vwap,
            line_disparity,
    ):
        self.ma_1.setData(line_ts, line_ma_1)
        self.vwap.setData(line_ts, line_vwap)
        self.disparity.setData(line_ts, line_disparity)

    def setTrendTitle(self, title: str):
        self.setTitle(trend_label_html(title, size=9))

    def setZeroLine(self, flag: bool):
        self.zero_line.setVisible(flag)

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

    def updateYAxisRange(self, flag: bool):
        self.zero_line.setVisible(flag)
        # self.plot_item.autoRange()
        self.plot_item.enableAutoRange(axis="x", enable=False)
        self.plot_item.setXRange(self.dict_ts["start"], self.dict_ts["end"])
        self.plot_item.enableAutoRange(axis="y", enable=True)
