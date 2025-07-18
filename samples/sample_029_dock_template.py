import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
)

from widgets.buttons import TradeButton
from widgets.containers import IndicatorBuySell, Widget
from widgets.docks import DockWidget
from widgets.labels import LCDIntWithTitle, LCDValueWithTitle
from widgets.layouts import GridLayout


class PanelTrading(Widget):
    def __init__(self):
        super().__init__()
        layout = GridLayout()
        self.setLayout(layout)

        row = 0
        # 建玉の売建
        self.sell = but_sell = TradeButton("sell")
        layout.addWidget(but_sell, row, 0)

        # 建玉の買建
        self.buy = but_buy = TradeButton("buy")
        layout.addWidget(but_buy, row, 1)

        row += 1
        # 建玉の売建（インジケータ）
        self.ind_sell = ind_sell = IndicatorBuySell()
        layout.addWidget(ind_sell, row, 0)

        # 建玉の買建（インジケータ）
        self.ind_buy = ind_buy = IndicatorBuySell()
        layout.addWidget(ind_buy, row, 1)

        row += 1
        # 建玉の返却
        self.repay = but_repay = TradeButton("repay")
        layout.addWidget(but_repay, row, 0, 1, 2)


class DockTemplate(DockWidget):
    def __init__(self, title: str):
        super().__init__(title)
        # 現在株価
        price = LCDValueWithTitle("現在株価")
        self.layout.addWidget(price)
        # 含み損益
        profit = LCDValueWithTitle("含み損益")
        self.layout.addWidget(profit)
        # 合計収益
        total = LCDValueWithTitle("合計収益")
        self.layout.addWidget(total)
        # EP 更新回数
        epupd = LCDIntWithTitle("EP 更新回数")
        self.layout.addWidget(epupd)

        # 取引用パネル
        trading = PanelTrading()
        self.layout.addWidget(trading)


class Example(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dock Template")
        ticker = "Ticker"
        dock = DockTemplate(ticker)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

        base = Widget()
        self.setCentralWidget(base)


def main():
    app = QApplication(sys.argv)
    win = Example()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
