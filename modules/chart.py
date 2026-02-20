import logging
from typing import Any

import pyqtgraph as pg
import pyqtgraph.exporters
from PySide6.QtCore import QMargins
from pyqtgraph import DateAxisItem

from funcs.plot import trend_label_html
from funcs.setting import get_trend_footer
from structs.res import AppRes
from widgets.misc import TickFont


class CustomYAxisItem(pg.AxisItem):
    def tickStrings(self, values: list[float], scale: float, spacing: float) -> list[str]:
        return [f"{value:6,.1f}" for value in values]


class TrendChart(pg.PlotWidget):
    COLOR_MA_1 = (0, 255, 0, 192)
    COLOR_VWAP = (255, 0, 192, 192)
    COLOR_GOLDEN = (255, 0, 204, 220)
    COLOR_DEAD = (0, 191, 255, 220)
    COLOR_DISPARITY = (255, 255, 0, 220)
    COLOR_EDGE = (128, 255, 0, 0)
    COLOR_FILL = (255, 255, 255, 96)
    COLOR_LAST_DOT = (0, 255, 0, 255)
    SIZE_LAST_DOT = 4

    def __init__(
            self,
            res: AppRes,
            dict_ts: dict[str, Any],
            dict_setting: dict[str, Any]
    ) -> None:
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
        # 移動 IQR バンド
        self.band_lower: pg.PlotDataItem = self.plot([], [], pen=pg.mkPen(self.COLOR_EDGE, width=1))
        self.band_lower.setZValue(40)
        self.band_upper: pg.PlotDataItem = self.plot([], [], pen=pg.mkPen(self.COLOR_EDGE, width=1))
        self.band_upper.setZValue(40)
        self.band = pg.FillBetweenItem(self.band_lower, self.band_upper, pg.mkBrush(self.COLOR_FILL))
        self.band.setZValue(30)
        self.addItem(self.band)

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

    def config_plot_item(self) -> None:
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

    def setCrossDead(self, x: float) -> None:
        self.vline_dead.setPos(x)

    def setCrossGolden(self, x: float) -> None:
        self.vline_golden.setPos(x)

    # def setLine(self, line_x, line_y):
    def setLine(self, line_x: list[float], line_y: list[float]) -> None:
        # トレンド線
        self.line.setData(tuple(line_x), tuple(line_y))

    def setDot(self, x: list[float], y: list[float]) -> None:
        # 最新値
        self.last_dot.setData(x, y)

    def setTechnicals(self, dict_lines: dict[str, list], visible: bool) -> None:
        data_ts = tuple(dict_lines["ts"])
        data_ma_1 = tuple(dict_lines["ma_1"])
        data_vwap = tuple(dict_lines["vwap"])
        data_disparity = tuple(dict_lines["disparity"])
        data_lower = tuple(dict_lines["lower"])
        data_upper = tuple(dict_lines["upper"])

        self.ma_1.setData(data_ts, data_ma_1)
        self.vwap.setData(data_ts, data_vwap)

        self.disparity.setData(data_ts, data_disparity)
        self.band_lower.setData(data_ts, data_lower)
        self.band_upper.setData(data_ts, data_upper)

        self.ma_1.setVisible(visible)
        self.vwap.setVisible(visible)
        self.disparity.setVisible(not visible)
        self.band_lower.setVisible(not visible)
        self.band_upper.setVisible(not visible)
        self.band.setVisible(not visible)

    def setTrendTitle(self, title: str) -> None:
        self.setTitle(trend_label_html(title, size=9))

    def setZeroLine(self, flag: bool) -> None:
        self.zero_line.setVisible(flag)

    def save(self, path_img: str) -> None:
        """
        チャートをイメージに保存
        https://pyqtgraph.readthedocs.io/en/latest/user_guide/exporting.html
        :param path_img:
        :return:
        """
        exporter = pg.exporters.ImageExporter(self.plot_item)
        exporter.export(path_img)
        self.logger.info(f"{__name__}: チャートが {path_img} に保存されました。")

    def updateYAxisRange(self, flag: bool) -> None:
        self.zero_line.setVisible(flag)
        # self.plot_item.autoRange()
        self.plot_item.enableAutoRange(axis="x", enable=False)
        self.plot_item.setXRange(self.dict_ts["start"], self.dict_ts["end"])
        self.plot_item.enableAutoRange(axis="y", enable=True)
