import logging
from typing import Any

import pandas as pd
import pyqtgraph as pg
import sys

from PySide6.QtCore import QMargins

from funcs.plot import trend_label_html
from funcs.setting import get_trend_footer
from structs.res import AppRes


class CustomYAxisItem1(pg.AxisItem):
    def tickStrings(self, values: list[float], scale: float, spacing: float) -> list[str]:
        return [f"{value: 6,.0f}" for value in values]


class CustomYAxisItem2(pg.AxisItem):
    def tickStrings(self, values: list[float], scale: float, spacing: float) -> list[str]:
        return [f"{value: 6,.2f}" for value in values]


class TrendCharts(pg.GraphicsLayoutWidget):
    COLOR_MA_1 = (0, 255, 0, 192)
    COLOR_VWAP = (255, 0, 192, 192)
    COLOR_GOLDEN = (255, 0, 204, 220)
    COLOR_DEAD = (0, 191, 255, 220)
    COLOR_EVEN = (255, 192, 0, 255)
    COLOR_LAST_DOT = (0, 255, 0, 255)
    COLOR_RSI = (255, 255, 0, 192)
    COLOR_ZERO = (255, 192, 192, 255)
    SIZE_LAST_DOT = 4

    def __init__(self, res: AppRes, dict_ts: dict[str, Any], dict_setting: dict[str, Any]):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.res = res
        self.dict_ts = dict_ts
        self.dict_setting = dict_setting

        # ---------------------------------------------------------------------
        # ウィンドウのサイズ制約（高さのみ）
        # self.setFixedHeight(400)
        self.setFixedHeight(res.trend_height)
        # self.setMidLineWidth(res.trend_width)
        self.setContentsMargins(QMargins(0, 0, 0, 0))

        # 価格チャート（上段）- CustomYAxisItem1 を適用
        self.plot_price = self.addPlot(
            row=0, col=0,
            axisItems={
                'left': CustomYAxisItem1(orientation='left'),
                'bottom': pg.DateAxisItem(orientation='bottom')
            }
        )
        self.plot_price.getAxis('bottom').setStyle(showValues=False)
        self.plot_price.setLabel('left', 'Price')

        # RSIチャート（下段）- CustomYAxisItem2 を適用
        self.plot_rsi = self.addPlot(
            row=1, col=0,
            axisItems={
                'left': CustomYAxisItem2(orientation='left'),
                'bottom': pg.DateAxisItem(orientation='bottom')
            }
        )
        self.plot_rsi.setLabel('left', 'RSI')
        # X軸を連動させる
        self.plot_rsi.setXLink(self.plot_price)
        # プロットの設定
        self._config_plot_items()

        # 移動平均線 MA1
        self.ma_1 = self.plot_price.plot(pen=pg.mkPen(self.COLOR_MA_1, width=1), name="MA1")
        self.ma_1.setZValue(60)

        # VWAP
        self.vwap = self.plot_price.plot(pen=pg.mkPen(self.COLOR_VWAP, width=1), name="VWAP")
        self.vwap.setZValue(50)

        # 損益分岐線
        # self.even_line = self.plot_price.addLine(y=0, pen=pg.mkPen(self.COLOR_EVEN, width=0.5))
        self.even_line = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(self.COLOR_EVEN, width=0.5))
        self.even_line.setZValue(0)
        self.even_line.setVisible(False)
        self.plot_price.addItem(self.even_line)

        # 最新値を示すドット
        self.last_dot = pg.ScatterPlotItem(
            size=self.SIZE_LAST_DOT,
            brush=pg.mkBrush(self.COLOR_LAST_DOT),
            pen=None
        )
        self.last_dot.setZValue(100)
        self.plot_price.addItem(self.last_dot)

        # クロス線（ゴールデン・クロス）
        self.vline_golden = pg.InfiniteLine(angle=90, pen=pg.mkPen(self.COLOR_GOLDEN, width=0.75))
        self.vline_golden.setZValue(20)
        self.plot_price.addItem(self.vline_golden)

        # クロス線（デッド・クロス）
        self.vline_dead = pg.InfiniteLine(angle=90, pen=pg.mkPen(self.COLOR_DEAD, width=0.75))
        self.vline_dead.setZValue(20)
        self.plot_price.addItem(self.vline_dead)

        # RSI
        self.rsi = self.plot_rsi.plot(pen=pg.mkPen(self.COLOR_RSI, width=1), name='RSI')

        # 基準線を追加
        self.plot_rsi.addLine(y=0.7, pen=pg.mkPen((255, 0, 255, 128), width=0.75))
        self.plot_rsi.addLine(y=0.5, pen=pg.mkPen((255, 255, 255, 96), width=0.75))
        self.plot_rsi.addLine(y=0.3, pen=pg.mkPen((0, 255, 255, 96), width=0.75))

    def _config_plot_items(self) -> None:
        self.ci.layout.setSpacing(0)
        self.ci.layout.setRowStretchFactor(0, 4)  # 上段は4
        self.ci.layout.setRowStretchFactor(1, 3)  # 下段は3

        # x軸範囲（ザラ場時間に固定）
        self.plot_price.setXRange(self.dict_ts["start"], self.dict_ts["end"])
        # y軸範囲（RSI）
        self.plot_rsi.setYRange(0, 1)

        # x軸ラベルをフッターとして扱う（日付と設定パラメータ）
        footer = get_trend_footer(self.dict_ts, self.dict_setting)
        self.plot_rsi.setLabel(axis="bottom", text=trend_label_html(footer, size=7))

        # x軸の余白を設定
        self.plot_rsi.getAxis('bottom').setHeight(26)

        for plot_item in [self.plot_price, self.plot_rsi]:
            # フォントの設定
            plot_item.getAxis('bottom').setStyle(tickFont=self.res.name_tick_font)
            plot_item.getAxis('left').setStyle(tickFont=self.res.name_tick_font)

            # グリッド
            plot_item.showGrid(x=True, y=True, alpha=0.5)

            # マウス操作無効化
            plot_item.setMouseEnabled(x=False, y=False)
            plot_item.setMenuEnabled(False)
            plot_item.hideButtons()

            # 高速化オプション
            plot_item.setClipToView(True)

    def setCrossDead(self, x: float) -> None:
        self.vline_dead.setPos(x)

    def setCrossGolden(self, x: float) -> None:
        self.vline_golden.setPos(x)

    def setDot(self, x: list[float], y: list[float]) -> None:
        # 最新値
        self.last_dot.setData(x, y)

    def setEvenLine(self, price: float):
        self.even_line.setPos(price)
        if price == 0.0:
            self.even_line.setVisible(False)
        else:
            self.even_line.setVisible(True)

    def setTechnicals(self, dict_lines: dict[str, list]) -> None:
        data_ts = dict_lines["ts"]
        data_ma_1 = dict_lines["ma_1"]
        data_vwap = dict_lines["vwap"]
        data_rsi = dict_lines["rsi"]

        self.ma_1.setData(data_ts, data_ma_1)
        self.vwap.setData(data_ts, data_vwap)
        self.rsi.setData(data_ts, data_rsi)

    def setTrendTitle(self, title: str) -> None:
        self.plot_price.setTitle(trend_label_html(title, size=9))

    def save(self, path_img: str) -> None:
        """
        チャートをイメージに保存
        :param path_img:
        :return:
        """
        exporter = pg.exporters.ImageExporter(self.plot_price)
        exporter.export(path_img)
        self.logger.info(f"{__name__}: チャートを {path_img} に保存しました。")

    def updateYAxisRange(self, flag: bool) -> None:
        self.zero_line.setVisible(flag)
        self.plot_item.enableAutoRange(axis="x", enable=False)
        self.plot_item.setXRange(self.dict_ts["start"], self.dict_ts["end"])
        self.plot_item.enableAutoRange(axis="y", enable=True)
