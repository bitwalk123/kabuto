from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import QDockWidget, QWidget

from structs.res import AppRes
from widgets.buttons import (
    ButtonBuy,
    ButtonRepay,
    ButtonSave,
    ButtonSell,
)
from widgets.containers import (
    Frame,
    PadH,
    Widget,
)
from widgets.labels import LCDNumber
from widgets.layouts import HBoxLayout, VBoxLayout


class DockTrader(QDockWidget):
    saveClicked = Signal()

    def __init__(self, res: AppRes):
        super().__init__()
        self.res = res

        self.setFeatures(
            QDockWidget.DockWidgetFeature.NoDockWidgetFeatures
        )
        self.setTitleBarWidget(QWidget())

        base = QWidget()
        self.setWidget(base)

        layout = VBoxLayout()
        layout.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        layout.setSpacing(2)

        base.setLayout(layout)

        # 現在株価表示
        self.lcd_price = lcd_price = LCDNumber(self)
        layout.addWidget(lcd_price)

        # 売買用ボタンの行
        row_buysell = Widget()
        layout.addWidget(row_buysell)
        layout_buysell = HBoxLayout()
        row_buysell.setLayout(layout_buysell)

        # 売掛ボタン
        but_sell = ButtonSell()
        layout_buysell.addWidget(but_sell)

        # 余白
        pad = PadH()
        layout_buysell.addWidget(pad)

        # 買掛ボタン
        but_buy = ButtonBuy()
        layout_buysell.addWidget(but_buy)

        # 含み損益表示
        self.lcd_profit = lcd_profit = LCDNumber(self)
        layout.addWidget(lcd_profit)

        # 建玉返済ボタン
        but_repay = ButtonRepay()
        layout.addWidget(but_repay)

        # 合計損益表示
        self.lcd_total = lcd_total = LCDNumber(self)
        layout.addWidget(lcd_total)

        # その他ツール用フレーム
        row_tool = Frame()
        layout.addWidget(row_tool)
        layout_tool = HBoxLayout()
        row_tool.setLayout(layout_tool)

        # 余白
        pad = PadH()
        layout_tool.addWidget(pad)

        # 画像保存ボタン
        but_save = ButtonSave()
        but_save.clicked.connect(self.on_save)
        layout_tool.addWidget(but_save)

    def on_save(self):
        # ---------------------------------
        # 🧿 保存ボタンがクリックされたことを通知
        # ---------------------------------
        self.saveClicked.emit()

    def setPrice(self, price: float):
        self.lcd_price.display(f"{price:.1f}")
