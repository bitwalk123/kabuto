from PySide6.QtCore import QMargins, Signal, Qt
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import (
    QFrame,
    QMainWindow,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QWidget, QAbstractScrollArea,
)

from structs.res import AppRes
from widgets.buttons import (
    ButtonSave,
    ButtonSetting,
    ToggleButtonAutoPilot,
    TradeButton,
)
from widgets.layouts import GridLayout, HBoxLayout


class Frame(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameStyle(
            QFrame.Shape.StyledPanel | QFrame.Shadow.Plain
        )
        self.setLineWidth(1)


class FrameSunken(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameStyle(
            QFrame.Shape.WinPanel | QFrame.Shadow.Sunken
        )
        self.setLineWidth(2)


class IndicatorBuySell(QFrame):
    def __init__(self):
        super().__init__()
        self.setContentsMargins(QMargins(0, 0, 0, 0))
        self.setFrameStyle(
            QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken
        )
        self.setLineWidth(2)
        self.setFixedHeight(5)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum
        )
        palette = self.palette()
        self.background_default = palette.color(QPalette.ColorRole.Window)
        # print(f"Default background color (RGB): {self.background_default.getRgb()}")

    def setDefault(self):
        self.setStyleSheet("")
        self.setPalette(self.background_default)

    def setBuy(self):
        self.setStyleSheet("QFrame{background-color: magenta;}")

    def setSell(self):
        self.setStyleSheet("QFrame{background-color: cyan;}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setContentsMargins(QMargins(0, 0, 0, 0))


class PadH(QWidget):
    def __init__(self):
        super().__init__()
        self.setContentsMargins(0, 0, 0, 0)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred
        )


class PadV(QWidget):
    def __init__(self):
        super().__init__()
        self.setContentsMargins(0, 0, 0, 0)
        self.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Expanding
        )


class TabWidget(QTabWidget):
    def __init__(self):
        super().__init__()
        self.setContentsMargins(QMargins(0, 0, 0, 0))
        self.setStyleSheet(
            """
            QTabWidget {
                font-family: monospace;
                font-size: 9pt;
            }
            """
        )


class Widget(QWidget):
    def __init__(self):
        super().__init__()
        self.setContentsMargins(QMargins(0, 0, 0, 0))


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


class PanelOption(QFrame):
    """
    トレーディング用オプションパネル
    """

    def __init__(self, res: AppRes):
        super().__init__()
        self.setFrameStyle(
            QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken
        )
        self.setLineWidth(1)
        layout = HBoxLayout()
        self.setLayout(layout)

        self.autopilot = but_autopilot = ToggleButtonAutoPilot(res)
        but_autopilot.setChecked(True)  # デフォルトで ON
        layout.addWidget(but_autopilot)

        hpad = PadH()
        layout.addWidget(hpad)

        self.save = but_save = ButtonSave(res)
        layout.addWidget(but_save)

        self.setting = but_setting = ButtonSetting(res)
        layout.addWidget(but_setting)

    def isAutoPilotEnabled(self) -> bool:
        return self.autopilot.isChecked()

    def setAutoPilotEnabled(self, state: bool = True):
        self.autopilot.setChecked(state)


class ScrollArea(QScrollArea):
    def __init__(self):
        super().__init__()
        self.setContentsMargins(QMargins(0, 0, 0, 0))
        self.setWidgetResizable(True)
        #self.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    """
    def sizeHint(self):
        if self.widget():
            return self.widget().sizeHint()
        return super().sizeHint()
    """
