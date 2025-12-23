import logging

import pyqtgraph as pg
import pyqtgraph.exporters
from PySide6.QtCore import QMargins
from pyqtgraph import DateAxisItem

from structs.res import AppRes
from widgets.misc import TickFont


class CustomYAxisItem(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        return [f"{value:6,.0f}" for value in values]


class TrendGraph(pg.PlotWidget):
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
        self.line: pg.PlotDataItem = self.plot([], [], pen=pg.mkPen(width=0.5))
        # 移動平均線 1
        self.ma_1: pg.PlotDataItem = self.plot([], [], pen=pg.mkPen("#0f0", width=0.75))
        # 移動平均線 2
        self.ma_2: pg.PlotDataItem = self.plot([], [], pen=pg.mkPen("#faf", width=1))

    def config_plot_item(self):
        """
        プロットアイテムの設定
        :param dict_ts:
        :param dict_setting:
        :return:
        """
        # ---------------------------------------------------------------------
        # x軸範囲（ザラ場時間に固定）
        self.plot_item.setXRange(self.dict_ts["start"], self.dict_ts["end"])
        # ---------------------------------------------------------------------
        # x軸ラベルをフッターとして扱う（日付と設定パラメータ）
        msg_footer = (
            f"DATE = {self.dict_ts['datetime_str_2']} / "
            f"PERIOD_MA_1 = {self.dict_setting['PERIOD_MA_1']} / "
            f"PERIOD_MA_2 = {self.dict_setting['PERIOD_MA_2']} / "
            f"PERIOD_MR = {self.dict_setting['PERIOD_MR']} / "
            f"THRESHOLD_MR = {self.dict_setting['THRESHOLD_MR']}"
        )
        self.plot_item.setLabel(
            axis="bottom",
            text=(
                '<span style="font-family: monospace; font-size: 7pt;">'
                f'{msg_footer}</span>'
            )
        )
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

    def setLine(self, line_x, line_y):
        self.line.setData(line_x, line_y)

    def setTechnicals(self, line_ts, line_ma_1, line_ma_2):
        self.ma_1.setData(line_ts, line_ma_1)
        self.ma_2.setData(line_ts, line_ma_2)

    def setTrendTitle(self, title: str):
        html = (
            '<span style="font-size: 9pt; font-family: monospace;">'
            f'{title}</span>'
        )
        self.setTitle(html)

    def save(self, path_img: str):
        """
        チャートをイメージに保存
        https://pyqtgraph.readthedocs.io/en/latest/user_guide/exporting.html
        :param path_img:
        :return:
        """
        # Create an exporter for the widget's scene
        # Note: For PlotWidget, you use widget.scene() or widget.plotItem
        exporter = pg.exporters.ImageExporter(self.plot_item)
        # Optional: Set export parameters like width/height (adjust as needed)
        # exporter.parameters()['width'] = 1000

        # Export to PNG file
        exporter.export(path_img)
        self.logger.info(f"{__name__}: チャートが {path_img} に保存されました。")
