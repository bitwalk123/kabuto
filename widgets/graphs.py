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
    COLOR_VWAP = (0, 255, 255, 192)
    COLOR_MA_1 = (0, 255, 0, 192)
    COLOR_MA_2 = (255, 0, 255, 255)
    COLOR_GOLDEN = (255, 0, 192, 160)
    COLOR_DEAD = (0, 192, 255, 128)
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
        font_mono = TickFont(self.res)
        axis_bottom.setStyle(tickFont=font_mono)
        axis_left.setStyle(tickFont=font_mono)

        super().__init__(
            axisItems={'bottom': axis_bottom, 'left': axis_left},
            enableMenu=False
        )

        # ---------------------------------------------------------------------
        # y軸の固定範囲対応
        self.vb = self.getViewBox()
        self.open_price = None
        self.fixed_range_active = False
        self.ratio_range = 0.005
        self.minY = None
        self.maxY = None

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
        # VWAP
        self.vwap: pg.PlotDataItem = self.plot([], [], pen=pg.mkPen(self.COLOR_VWAP, width=0.5))
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

        self.vline_golden = pg.InfiniteLine(angle=90, pen=pg.mkPen(self.COLOR_GOLDEN, width=1))
        self.addItem(self.vline_golden)

        self.vline_dead = pg.InfiniteLine(angle=90, pen=pg.mkPen(self.COLOR_DEAD, width=1))
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
        # line_x, line_y が空でないことが保証されている
        t = line_x[-1]
        price = line_y[-1]

        # 最初のy軸はは固定レンジ
        if self.open_price is None:
            self.yrange_set_fixed(price)

        # y軸が固定レンジ中であればチェック
        if self.fixed_range_active:
            self.yrange_check(price)

        # トレンド線
        self.line.setData(line_x, line_y)
        # 最新値
        self.last_dot.setData([t], [price])

    def setTechnicals(self, line_ts, line_vwap, line_ma_1, line_ma_2):
        self.vwap.setData(line_ts, line_vwap)
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

    def yrange_check(self, price):
        if price < self.minY or self.maxY < price:
            # 範囲を超えた → オートスケールに切り替え
            self.vb.enableAutoRange(axis=pg.ViewBox.YAxis)
            self.fixed_range_active = False

    def yrange_set_fixed(self, price: float):
        self.open_price = price
        width = price * self.ratio_range
        self.minY = price - width
        self.maxY = price + width
        # 固定レンジを設定
        self.vb.disableAutoRange(axis=pg.ViewBox.YAxis)
        self.vb.setYRange(self.minY, self.maxY)
        self.fixed_range_active = True
