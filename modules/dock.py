import logging

from PySide6.QtCore import Signal, QMargins

from modules.panel import PanelOption, PanelTrading
from structs.res import AppRes
from widgets.dialogs import DlgRepair
from widgets.docks import DockWidget
from widgets.labels import LCDValueWithTitle, LabelSmall


class DockTrader(DockWidget):
    clickedBuy = Signal(str, float, str, bool)
    clickedSell = Signal(str, float, str, bool)
    clickedRepay = Signal(str, float, str, bool)
    changedDisparityState = Signal(bool)
    clickedSave = Signal()

    def __init__(self, res: AppRes, code: str):
        super().__init__(code)
        self.logger = logging.getLogger(__name__)
        self.res = res
        self.code = code

        """
        自動オペレーション用フラグ
        マウスで売買ボタンをクリックしたか、
        エージェントが売買シグナルを出したのかを
        区別するためのフラグ
        """
        self.auto = False

        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        #  UI
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        self.setContentsMargins(QMargins(5, 2, 5, 2))

        # 現在株価（表示）
        self.price = price = LCDValueWithTitle("現在株価")
        self.layout.addWidget(price)
        # 含み損益（表示）
        self.profit = profit = LCDValueWithTitle("含み損益")
        self.layout.addWidget(profit)
        # 合計収益（表示）
        self.total = total = LCDValueWithTitle("合計収益")
        self.layout.addWidget(total)

        # ---------------------------------------------------------------------
        # 取引用パネル
        # ---------------------------------------------------------------------
        self.trading = trading = PanelTrading()
        trading.clickedBuy.connect(self.on_buy)
        trading.clickedSell.connect(self.on_sell)
        trading.clickedRepay.connect(self.on_repay)
        self.layout.addWidget(trading)

        # ---------------------------------------------------------------------
        # オプションパネル
        # ---------------------------------------------------------------------
        # 「乖離度」用ラベル
        lab_disparity = LabelSmall("乖離度")
        self.layout.addWidget(lab_disparity)
        # 「オプション」用パネル
        self.option = option = PanelOption(res, code)
        option.clickedSave.connect(self.on_save)
        option.clickedRepair.connect(self.on_repair)
        option.changedDisparity.connect(self.disparity_changed)
        self.layout.addWidget(option)

    def forceRepay(self):
        """
        強制返済（取引終了時）
        :return:
        """
        if self.doRepay():
            self.logger.info(f"{__name__}: '{self.code}'の強制返済をしました。")

    def on_buy(self):
        """
        買建ボタンがクリックされた時の処理
        :return:
        """
        note = ""
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 買建ボタンがクリックされたことを通知
        self.clickedBuy.emit(
            self.code, self.price.getValue(), note, self.auto
        )
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.auto = False

    def on_sell(self):
        """
        売建ボタンがクリックされた時の処理
        :return:
        """
        note = ""
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 売建ボタンがクリックされたことを通知
        self.clickedSell.emit(
            self.code, self.price.getValue(), note, self.auto
        )
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.auto = False

    def on_repay(self):
        """
        返済ボタンがクリックされた時の処理
        :return:
        """
        note = ""
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 返済ボタンがクリックされたことを通知
        self.clickedRepay.emit(
            self.code, self.price.getValue(), note, self.auto
        )
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.auto = False

    def isDisparityChecked(self) -> bool:
        return self.option.disparity.isEnabled()

    def disparity_changed(self, status: bool):
        """for statusChanged signal
        """
        # print('Switch is', status)
        self.changedDisparityState.emit(status)

    def on_save(self):
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 保存ボタンがクリックされたことを通知
        self.clickedSave.emit()
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def on_repair(self):
        dlg = DlgRepair(self.res)
        if dlg.exec():
            flag = dlg.getStatus()
            self.trading.switch_activate(flag)
        else:
            return

    def setPrice(self, price: float):
        """
        現在株価を表示
        :param price:
        :return:
        """
        self.price.setValue(price)

    def setProfit(self, profit: float):
        """
        現在の含み益を表示
        :param profit:
        :return:
        """
        self.profit.setValue(profit)

    def setTotal(self, total: float):
        """
        現在の損益合計を表示
        :param total:
        :return:
        """
        self.total.setValue(total)

    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
    # Agent からのアクション
    # 手動でボタンをクリックした時と区別できるようにする
    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
    def doBuy(self) -> bool:
        """
        「買建」ボタンをクリックして建玉を売る。
        :return:
        """
        if self.trading.buy.isEnabled():
            self.auto = True
            self.trading.buy.animateClick()
            return True
        else:
            self.auto = False
            return False

    def doSell(self) -> bool:
        """
        「売建」ボタンをクリックして建玉を売る。
        :return:
        """
        if self.trading.sell.isEnabled():
            self.auto = True
            self.trading.sell.animateClick()
            return True
        else:
            self.auto = False
            return False

    def doRepay(self) -> bool:
        """
        「返済」ボタンをクリックして建玉を売る。
        :return:
        """
        if self.trading.repay.isEnabled():
            self.auto = True
            self.trading.repay.animateClick()
            return True
        else:
            self.auto = False
            return False

    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
    # （実売買移行用）
    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
    def receive_result(self, status: bool):
        self.trading.receive_result(status)
