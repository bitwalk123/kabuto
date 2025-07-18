from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow, QPushButton

from structs.res import AppRes
from widgets.containers import PanelOption, PanelTrading, Widget
from widgets.docks import DockWidget
from widgets.labels import LCDValueWithTitle, LCDIntWithTitle
from widgets.layouts import VBoxLayout


class DockTemplate(DockWidget):
    def __init__(self, res: AppRes, title: str):
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

        # オプションパネル
        self.option = option = PanelOption(res)
        self.layout.addWidget(option)

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
        res = AppRes()

        self.setWindowTitle("Dock Template")
        ticker = "Ticker"
        self.dock = dock = DockTemplate(res, ticker)
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
