import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow, QPushButton,
)

from widgets.buttons import ButtonBuy, ButtonSell, ButtonRepay
from widgets.containers import Widget, PadH
from widgets.docks import DockWidget
from widgets.labels import LCDValueWithTitle, LCDIntWithTitle
from widgets.layouts import VBoxLayout, HBoxLayout


class TradeButton(QPushButton):
    def __init__(self):
        super().__init__()
        font = QFont()
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPointSize(9)
        self.setFont(font)


class PanelTrading(Widget):
    def __init__(self):
        super().__init__()
        layout = VBoxLayout()
        self.setLayout(layout)

        # 一行用レイアウト
        layout_row = HBoxLayout()
        layout.addLayout(layout_row)

        # 建玉の売建
        self.sell = but_sell = TradeButton()
        but_sell.setText("売　建")
        layout_row.addWidget(but_sell)

        # 余白
        pad = PadH()
        layout_row.addWidget(pad)

        # 建玉の買建
        self.buy = but_buy = TradeButton()
        but_buy.setText("買　建")
        layout_row.addWidget(but_buy)

        # 建玉の返却
        self.repay = but_repay = TradeButton()
        but_repay.setText("返　　却")
        layout.addWidget(but_repay)


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
