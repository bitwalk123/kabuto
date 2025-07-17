import logging

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import QDockWidget

from structs.res import AppRes
from widgets.buttons import (
    ButtonBuy,
    ButtonConf,
    ButtonRepay,
    ButtonSave,
    ButtonSell,
    ButtonSemiAuto,
    ToggleButtonAutoPilot,
)
from widgets.containers import (
    Frame,
    PadH,
    Widget,
)
from widgets.labels import (
    LabelRightSmall,
    LabelSmall,
    LCDInt,
    LCDNumber,
)
from widgets.layouts import HBoxLayout, VBoxLayout


class DockWidget(QDockWidget):
    def __init__(self, title: str = ""):
        super().__init__()

        self.setFeatures(
            QDockWidget.DockWidgetFeature.NoDockWidgetFeatures
        )
        self.setTitleBarWidget(LabelRightSmall(title))

        base = Widget()
        self.setWidget(base)

        self.layout = layout = VBoxLayout()
        layout.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        layout.setSpacing(2)
        base.setLayout(layout)


class DockTrader(QDockWidget):
    clickedConf = Signal(str)
    clickedSave = Signal(str)
    clickedBuy = Signal(str, float, str)
    clickedSell = Signal(str, float, str)
    clickedRepay = Signal(str, float, str)

    def __init__(self, res: AppRes, ticker: str):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.res = res
        self.ticker = ticker
        self.trend: int = 0

        self.setFeatures(
            QDockWidget.DockWidgetFeature.NoDockWidgetFeatures
        )
        self.setTitleBarWidget(Widget())

        base = Widget()
        self.setWidget(base)

        layout = VBoxLayout()
        layout.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        layout.setSpacing(2)
        base.setLayout(layout)

        # 現在株価表示
        lab_title_price = LabelSmall("現在株価")
        layout.addWidget(lab_title_price)
        self.lcd_price = lcd_price = LCDNumber(self)
        layout.addWidget(lcd_price)

        # 売買用ボタンの行
        row_buysell = Widget()
        layout.addWidget(row_buysell)
        layout_buysell = HBoxLayout()
        row_buysell.setLayout(layout_buysell)

        # 売掛ボタン
        self.but_sell = but_sell = ButtonSell()
        # ----------------------------------------------------------
        # 意図的に clicked シグナルを使用している
        # 買建、売建、返済ボタンの組み合わせで Enable, Disable をカスタマイズ
        # ----------------------------------------------------------
        but_sell.clicked.connect(self.on_sell)
        layout_buysell.addWidget(but_sell)

        # 余白
        pad = PadH()
        layout_buysell.addWidget(pad)

        # 買掛ボタン
        self.but_buy = but_buy = ButtonBuy()
        # ----------------------------------------------------------
        # 意図的に clicked シグナルを使用している
        # 買建、売建、返済ボタンの組み合わせで Enable, Disable をカスタマイズ
        # ----------------------------------------------------------
        but_buy.clicked.connect(self.on_buy)
        layout_buysell.addWidget(but_buy)

        # 含み損益表示
        lab_title_profit = LabelSmall("含み損益")
        layout.addWidget(lab_title_profit)
        self.lcd_profit = lcd_profit = LCDNumber(self)
        layout.addWidget(lcd_profit)

        # 建玉返済ボタン
        self.but_repay = but_repay = ButtonRepay()
        self.but_repay.setDisabled(True)
        # ----------------------------------------------------------
        # 意図的に clicked シグナルを使用している
        # 買建、売建、返済ボタンの組み合わせで Enable, Disable をカスタマイズ
        # ----------------------------------------------------------
        but_repay.clicked.connect(self.on_repay)
        layout.addWidget(but_repay)

        # EP 更新回数
        lab_epupd = LabelSmall("EP 更新回数")
        layout.addWidget(lab_epupd)
        self.lcd_epupd = lcd_epupd = LCDInt(self)
        layout.addWidget(lcd_epupd)

        # セミオートボタン（利確・損切のために手動で返済できる）
        self.semi_auto = but_semi_auto = ButtonSemiAuto()
        but_semi_auto.setEnabled(False)
        # ----------------------------------
        # 通常通り toggled シグナルを使用している
        # ----------------------------------
        but_semi_auto.toggled.connect(self.on_toggled_semi_auto)
        layout.addWidget(but_semi_auto)

        # 合計損益表示
        lab_title_total = LabelSmall("合計損益")
        layout.addWidget(lab_title_total)
        self.lcd_total = lcd_total = LCDNumber(self)
        layout.addWidget(lcd_total)

        # その他ツール用フレーム
        row_tool = Frame()
        layout.addWidget(row_tool)
        layout_tool = HBoxLayout()
        row_tool.setLayout(layout_tool)

        # オート用トグルボタン
        self.autopilot = but_autopilot = ToggleButtonAutoPilot(res)
        but_autopilot.setChecked(True)  # デフォルトで ON
        layout_tool.addWidget(but_autopilot)

        # 余白
        pad = PadH()
        layout_tool.addWidget(pad)

        # 画像保存ボタン
        but_save = ButtonSave()
        but_save.setToolTip("チャート保存")
        but_save.clicked.connect(self.on_save)
        layout_tool.addWidget(but_save)

        # 設定ボタン
        but_conf = ButtonConf()
        but_conf.setToolTip("設定")
        but_conf.clicked.connect(self.on_conf)
        layout_tool.addWidget(but_conf)

    def actSellBuy(self):
        """
        買建あるいは売建ボタンをクリックしたら Disable にし、
        返済ボタンを Enable にしてナンピン売買ができないようにする。
        :return:
        """
        self.but_buy.setEnabled(False)
        self.but_sell.setEnabled(False)
        self.but_repay.setEnabled(True)

    def actRepay(self):
        """
        返済ボタンをクリックしたら Disable にして、
        買建および売建ボタンを Enable にする。
        :return:
        """
        self.but_buy.setEnabled(True)
        self.but_buy.setChecked(False)
        self.but_sell.setEnabled(True)
        self.but_sell.setChecked(False)
        self.but_repay.setEnabled(False)

    def finishAutoTrade(self):
        if self.autopilot.isChecked():
            self.autopilot.setChecked(False)
            self.logger.info(
                f"{__name__} {self.ticker} のオートボタンのチェックを外しました。"
            )
            if self.semi_auto.isEnabled():
                if self.semi_auto.isChecked():
                    self.semi_auto.setChecked(False)
                    self.logger.info(
                        f"{__name__} {self.ticker} のセミオートボタンのチェックを外しました。"
                    )

    def getPrice(self) -> float:
        return self.lcd_price.value()

    def on_buy(self, note: str = ""):
        # -------------------------------------------
        # 🧿 買建ボタンがクリックされたことを通知
        self.clickedBuy.emit(
            self.ticker, self.getPrice(), note
        )
        # -------------------------------------------
        self.actSellBuy()

    def on_conf(self):
        # ---------------------------------
        # 🧿 設定ボタンがクリックされたことを通知
        self.clickedConf.emit(self.ticker)
        # ---------------------------------

    def on_repay(self, note: str = ""):
        # -------------------------------------------
        # 🧿 返済ボタンがクリックされたことを通知
        self.clickedRepay.emit(
            self.ticker, self.getPrice(), note
        )
        # -------------------------------------------
        self.actRepay()

    def on_save(self):
        # ---------------------------------
        # 🧿 保存ボタンがクリックされたことを通知
        self.clickedSave.emit(self.ticker)
        # ---------------------------------

    def on_sell(self, note: str = ""):
        # -------------------------------------------
        # 🧿 売建ボタンがクリックされたことを通知
        self.clickedSell.emit(
            self.ticker, self.getPrice(), note
        )
        # -------------------------------------------
        self.actSellBuy()

    def on_toggled_semi_auto(self, state: bool):
        if state:
            self.position_open_by_trend()
        else:
            self.position_close()

    def position_open_by_trend(self):
        if self.trend > 0:
            note = "買建（トレンド追従）"
            self.on_buy(note)
        elif self.trend < 0:
            note = "売建（トレンド追従）"
            self.on_sell(note)
        self.but_repay.setEnabled(False)

    def position_close(self):
        note = "返済"
        self.on_repay(note)

    def position_close_auto(self):
        note = "トレンド反転→返済（オート）"
        self.on_repay(note)
        self.semi_auto.setChecked(False)

    def setEPUpd(self, epupd: int):
        if epupd > 0:
            self.semi_auto.setEnabled(True)
            if epupd == 1 and self.autopilot.isChecked():
                if not self.semi_auto.isChecked():
                    self.semi_auto.setChecked(True)
                    self.logger.info(
                        f"{__name__} {self.ticker} のセミオートボタンをチェックしました。"
                    )
        else:
            self.semi_auto.setEnabled(False)

        self.lcd_epupd.display(f"{epupd}")

    def setPrice(self, price: float):
        self.lcd_price.display(f"{price:.1f}")

    def setProfit(self, profit: float):
        self.lcd_profit.display(f"{profit:.1f}")

    def setTotal(self, total: float):
        self.lcd_total.display(f"{total:.1f}")

    def setTrend(self, trend: int, epupd: int):
        if self.trend != trend:
            if self.semi_auto.isChecked():
                self.semi_auto.setChecked(False)
                self.logger.info(
                    f"{__name__} {self.ticker} のセミオートボタンのチェックを外しました。"
                )

        self.trend = trend
        self.setEPUpd(epupd)
