from PySide6.QtCore import Signal

from widgets.buttons import (
    TradeButton,
)
from widgets.containers import (
    IndicatorBuySell,
    Widget,
)
from widgets.layouts import GridLayout


class PanelTrading(Widget):
    """
    トレーディング用パネル
    固定株数でナンピンしない取引を前提にしている
    """
    clickedBuy = Signal()
    clickedSell = Signal()
    clickedRepay = Signal()

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
        # ---------------------------------------------------------------------
        # 🧿 買建ボタンがクリックされたことを通知
        self.clickedBuy.emit()
        # ---------------------------------------------------------------------
        self.position_open()
        self.ind_buy.setBuy()

    def on_sell(self):
        # ---------------------------------------------------------------------
        # 🧿 売建ボタンがクリックされたことを通知
        self.clickedSell.emit()
        # ---------------------------------------------------------------------
        self.position_open()
        self.ind_sell.setSell()

    def on_repay(self):
        # ---------------------------------------------------------------------
        # 🧿 返却ボタンがクリックされたことを通知
        self.clickedRepay.emit()
        # ---------------------------------------------------------------------
        self.position_close()
        self.ind_buy.setDefault()
        self.ind_sell.setDefault()
