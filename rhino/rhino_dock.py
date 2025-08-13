import logging

from PySide6.QtCore import Signal

from rhino.rhino_pacman import PacMan
from modules.panel import PanelOption, PanelTrading
from rhino.rhino_psar import PSARObject
from rhino.rhino_ticker import Ticker
from structs.app_enum import PositionType
from structs.res import AppRes
from widgets.docks import DockWidget
from widgets.labels import LCDIntWithTitle, LCDValueWithTitle


class DockTrader(DockWidget):
    clickedBuy = Signal(str, float, str)
    clickedSell = Signal(str, float, str)
    clickedRepay = Signal(str, float, str)
    notifyNewPSARParams = Signal(str, dict)

    def __init__(self, res: AppRes, code: str):
        super().__init__(code)
        self.logger = logging.getLogger(__name__)
        self.code = code
        self.pacman = PacMan()  # 売買判定用インスタンス
        self.ticker: Ticker | None = None

        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        #  UI
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_

        # RSS で売買するかを切り替えるスイッチ
        #row_swicth = SwitchRSS()
        #self.layout.addWidget(row_swicth)

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

        # EP 更新回数（表示）
        self.epupd = epupd = LCDIntWithTitle("EP 更新回数")
        self.layout.addWidget(epupd)

        # ---------------------------------------------------------------------
        # オプションパネル
        # ---------------------------------------------------------------------
        self.option = option = PanelOption(res, code)
        option.requestPSARParams.connect(self.request_psar_params)
        option.requestDefaultPSARParams.connect(self.request_default_psar_params)
        option.notifyNewPSARParams.connect(self.notify_new_psar_params)
        option.requestOEStatusChange.connect(self.request_Over_drive_status_change)
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

    def forceStopAutoPilot(self):
        """
        強制返済
        :return:
        """
        if self.doRepay():
            self.logger.info(f"{__name__}: '{self.code}'の強制返済をしました。")
        if self.option.isAutoPilotEnabled():
            self.option.setAutoPilotEnabled(False)
            self.logger.info(f"{__name__}: '{self.code}'の Autopilot をオフにしました。")

    def isOverDriveEnabled(self) -> bool:
        """
        Over Drive ボタンの状態を返す
        :return:
        """
        return self.option.isOverDriveEnabled()

    def notify_new_psar_params(self, dict_psar: dict):
        """
        新しいパラメータを通知
        :param dict_psar:
        :return:
        """
        self.notifyNewPSARParams.emit(self.code, dict_psar)

    def on_buy(self):
        """
        買建ボタンがクリックされた時の処理
        :return:
        """
        note = ""
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 買建ボタンがクリックされたことを通知
        self.clickedBuy.emit(
            self.code, self.price.getValue(), note
        )
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def on_repay(self):
        """
        返済ボタンがクリックされた時の処理
        :return:
        """
        note = ""
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 返済ボタンがクリックされたことを通知
        self.clickedRepay.emit(
            self.code, self.price.getValue(), note
        )
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def on_sell(self):
        """
        売建ボタンがクリックされた時の処理
        :return:
        """
        note = ""
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 売建ボタンがクリックされたことを通知
        self.clickedSell.emit(
            self.code, self.price.getValue(), note
        )
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def receive_default_psar_params(self, dict_default_psar: dict):
        """
        デフォルトのパラメータ値を取得した時の処理
        :param dict_default_psar:
        :return:
        """
        self.option.setDefaultPSARParams(dict_default_psar)

    def receive_psar_params(self, dict_psar: dict):
        """
        現在のパラメータ値を取得した時の処理
        :param dict_psar:
        :return:
        """
        self.option.showTradeConfig(dict_psar)

    def request_default_psar_params(self):
        """
        デフォルトのパラメータ値の要求
        :return:
        """
        if self.ticker is not None:
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # 🧿 デフォルトのパラメータ値の要求シグナル
            self.ticker.requestDefaultPSARParams.emit()
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def request_Over_drive_status_change(self, state: bool):
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 Over Drive の状態変更の要求シグナル
        self.ticker.requestOEStatusChange.emit(state)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def request_psar_params(self):
        if self.ticker is not None:
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # 🧿 現在のパラメータ値の要求シグナル
            self.ticker.requestPSARParams.emit()
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def setEPUpd(self, epupd: int):
        """
        EP更新回数を表示
        :param epupd:
        :return:
        """
        self.epupd.setValue(epupd)

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

    def set_over_drive_enabled(self, state: bool):
        """
        Over Drive ボタンの状態を設定
        :param state:
        :return:
        """
        self.option.setOverDriveEnabled(state)

    def setTrend(self, ret: PSARObject):
        """
        Parabolic SAR のトレンドに応じた売買処理
        :param ret:
        :return:
        """
        self.setEPUpd(ret.epupd)
        if self.option.isAutoPilotEnabled():
            ptype: PositionType = self.pacman.setTrend(ret)
            if ptype == PositionType.BUY:
                self.doBuy()
            elif ptype == PositionType.SELL:
                self.doSell()
            elif ptype == PositionType.REPAY:
                self.doRepay()
            else:
                pass

    def setTicker(self, ticker: Ticker):
        """
        Ticker インスタンスの保持とスロットの設定
        :param ticker:
        :return:
        """
        self.ticker = ticker
        ticker.worker.notifyPSARParams.connect(self.receive_psar_params)
        ticker.worker.notifyDefaultPSARParams.connect(self.receive_default_psar_params)
        ticker.worker.notifyODStatusChanged.connect(self.set_over_drive_enabled)
