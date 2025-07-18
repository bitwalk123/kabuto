import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow, QPushButton,
)

from widgets.buttons import TradeButton
from widgets.containers import IndicatorBuySell, Widget
from widgets.docks import DockWidget
from widgets.labels import LCDIntWithTitle, LCDValueWithTitle
from widgets.layouts import GridLayout, VBoxLayout


class PanelTrading(Widget):
    def __init__(self):
        super().__init__()
        layout = GridLayout()
        self.setLayout(layout)

        row = 0
        # 建玉の売建（インジケータ）
        self.ind_sell = ind_sell = IndicatorBuySell()
        layout.addWidget(ind_sell, row, 0)

        # 建玉の買建（インジケータ）
        self.ind_buy = ind_buy = IndicatorBuySell()
        layout.addWidget(ind_buy, row, 1)

        row += 1
        # 建玉の売建
        self.sell = but_sell = TradeButton("sell")
        but_sell.clicked.connect(self.on_sell)
        layout.addWidget(but_sell, row, 0)

        # 建玉の買建
        self.buy = but_buy = TradeButton("buy")
        but_buy.clicked.connect(self.on_buy)
        layout.addWidget(but_buy, row, 1)

        row += 1
        # 建玉の返却
        self.repay = but_repay = TradeButton("repay")
        but_repay.clicked.connect(self.on_repay)
        layout.addWidget(but_repay, row, 0, 1, 2)

        # 初期状態ではポジション無し
        self.position_close()

    def position_close(self):
        self.sell.setEnabled(True)
        self.buy.setEnabled(True)
        self.repay.setDisabled(True)

    def position_open(self):
        self.sell.setDisabled(True)
        self.buy.setDisabled(True)
        self.repay.setEnabled(True)

    def on_buy(self):
        self.position_open()
        self.ind_buy.setBuy()

    def on_sell(self):
        self.position_open()
        self.ind_sell.setSell()

    def on_repay(self):
        self.position_close()
        self.ind_buy.setDefault()
        self.ind_sell.setDefault()


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
        self.trading = trading = PanelTrading()
        self.layout.addWidget(trading)

    def doBuy(self) -> bool:
        """
        「買建」ボタンをクリックして建玉を売る。
        :return:
        """
        if self.trading.buy.isEnabled():
            self.trading.buy.animateClick()
            return True
        else:
            return False

    def doSell(self) -> bool:
        """
        「売建」ボタンをクリックして建玉を売る。
        :return:
        """
        if self.trading.sell.isEnabled():
            self.trading.sell.animateClick()
            return True
        else:
            return False

    def doRepay(self) -> bool:
        """
        「返済」ボタンをクリックして建玉を売る。
        :return:
        """
        if self.trading.repay.isEnabled():
            self.trading.repay.animateClick()
            return True
        else:
            return False


class Example(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dock Template")
        ticker = "Ticker"
        self.dock = dock = DockTemplate(ticker)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

        base = Widget()
        self.setCentralWidget(base)

        layout = VBoxLayout()
        layout.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        base.setLayout(layout)

        but_test_buy = QPushButton("買建")
        but_test_buy.clicked.connect(self.click_test_buy)
        layout.addWidget(but_test_buy)

        but_test_sell = QPushButton("売建")
        but_test_sell.clicked.connect(self.click_test_sell)
        layout.addWidget(but_test_sell)

        but_test_repay = QPushButton("返却")
        but_test_repay.clicked.connect(self.click_test_repay)
        layout.addWidget(but_test_repay)

    def click_test_buy(self):
        if not self.dock.doBuy():
            print("建玉を買建できませんでした。")

    def click_test_sell(self):
        if not self.dock.doSell():
            print("建玉を売建できませんでした。")

    def click_test_repay(self):
        if not self.dock.doRepay():
            print("建玉を返却できませんでした。")


def main():
    app = QApplication(sys.argv)
    win = Example()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
