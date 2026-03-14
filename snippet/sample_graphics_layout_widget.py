import pandas as pd
from PySide6.QtCore import QMargins, Qt
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
import pyqtgraph as pg
import sys


class CustomYAxisItem1(pg.AxisItem):
    def tickStrings(self, values: list[float], scale: float, spacing: float) -> list[str]:
        return [f"{value: 6,.0f}" for value in values]


class CustomYAxisItem2(pg.AxisItem):
    def tickStrings(self, values: list[float], scale: float, spacing: float) -> list[str]:
        return [f"{value: 6,.0f}" for value in values]


class TrendCharts(pg.GraphicsLayoutWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(600, 400)

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

        self.curve_ma_1 = self.plot_price.plot(pen=pg.mkPen((0, 255, 0), width=0.75), name='Price')
        self.curve_vwap = self.plot_price.plot(pen=pg.mkPen((255, 0, 255), width=0.75), name='VWAP')

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

        self.curve_rsi = self.plot_rsi.plot(pen=pg.mkPen((255, 255, 0), width=0.75), name='RSI')

        # プロットの設定
        self._config_plot_items()

    def _config_plot_items(self) -> None:
        path_font = "fonts/RictyDiminished-Regular.ttf"
        font_id = QFontDatabase.addApplicationFont(path_font)
        font = QFontDatabase.applicationFontFamilies(font_id)[0]

        self.ci.layout.setSpacing(0)
        self.ci.layout.setRowStretchFactor(0, 4)  # 上段は4
        self.ci.layout.setRowStretchFactor(1, 3)  # 下段は3

        for plot_item in [self.plot_price, self.plot_rsi]:
            # フォントの設定
            plot_item.getAxis('bottom').setStyle(tickFont=font)
            plot_item.getAxis('left').setStyle(tickFont=font)

            # グリッド
            plot_item.showGrid(x=True, y=True, alpha=0.5)

            # マウス操作無効化
            plot_item.setMouseEnabled(x=False, y=False)
            plot_item.setMenuEnabled(False)
            plot_item.hideButtons()

            # 高速化オプション
            plot_item.setClipToView(True)

    def update_data(self, times: list, ma_1: list, vwap, rsi: list):
        """データ更新"""
        self.curve_ma_1.setData(times, ma_1)
        self.curve_vwap.setData(times, vwap)
        self.curve_rsi.setData(times, rsi)

    def setTrendTitle(self, title: str) -> None:
        self.plot_price.setTitle(title)


class SampleCharts(QMainWindow):
    def __init__(self):
        super().__init__()
        # サンプル・データ
        file_csv = "sample_rsi.zip"
        df = pd.read_csv(file_csv, index_col=0)
        ts = list(df["ts"])
        ma1 = list(df["ma1"])
        vwap = list(df["vwap"])
        rsi = list(df["rsi"])

        self.setWindowTitle("pg.GraphicsLayoutWidget を利用したチャート・サンプル")
        base = QWidget()
        self.setCentralWidget(base)
        layout = QVBoxLayout(base)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        layout.setContentsMargins(QMargins(0, 0, 0, 0))

        # GraphicsLayoutWidget を作成
        self.trends = trends = TrendCharts()
        trends.setTrendTitle("銘柄名 (銘柄コード)")
        layout.addWidget(trends)
        trends.update_data(ts, ma1, vwap, rsi)


def main() -> None:
    app = QApplication(sys.argv)
    win = SampleCharts()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
