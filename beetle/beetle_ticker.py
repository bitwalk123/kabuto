"""
Ticker 毎のデータ処理クラス（銘柄スレッド・クラス）
機能スコープ
1. Realtime PSAR
"""
import logging
from collections import deque

from PySide6.QtCore import (
    QObject,
    QThread,
    Signal,
    Slot,
)

from beetle.beetle_psar import PSARObject, RealtimePSAR


class TickerWorker(QObject):
    # Parabolic SAR の情報を通知
    notifyPSAR = Signal(str, float, PSARObject)

    def __init__(self, ticker, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.ticker = ticker
        self.psar = RealtimePSAR()
        self.factor_median = 3  # メディアン値を算出するデータ点数
        self.deque_median = deque(maxlen=self.factor_median)
        self.period = 60
        self.deque_mr = deque(maxlen=self.period)

    @Slot(float, float)
    def addPrice4PSAR(self, x, y):
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # Realtime PSAR の算出
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        ret: PSARObject = self.psar.add(y)
        # ---------------------------------------------
        # 🧿 Parabolic SAR の情報を通知
        self.notifyPSAR.emit(self.ticker, x, ret)
        # ---------------------------------------------


# QThreadを継承した銘柄スレッドクラス
class ThreadTicker(QThread):
    """
    各銘柄専用のスレッド。
    TickerWorkerオブジェクトをこのスレッドに移動させる。
    """
    notifyNewPrice = Signal(float, float)

    # このスレッドが開始されたことを通知するシグナル（デバッグ用など）
    threadReady = Signal(str)

    def __init__(self, ticker, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.ticker = ticker
        self.worker = TickerWorker(ticker)
        self.worker.moveToThread(self)  # TickerWorkerをこのQThreadに移動

        # スレッド開始時にworkerの準備完了を通知 (必要であれば)
        self.started.connect(self.thread_ready)

        # メインスレッドからワーカースレッドへ新たな株価情報を通知
        self.notifyNewPrice.connect(self.worker.addPrice4PSAR)

    def thread_ready(self):
        self.threadReady.emit(self.ticker)

    def run(self):
        """
        このスレッドのイベントループを開始する。
        これがなければ、スレッドはすぐに終了してしまう。
        """
        self.logger.info(
            f"{__name__} ThreadTicker for {self.ticker}: run() method started. Entering event loop..."
        )
        self.exec()  # イベントループを開始
        self.logger.info(
            f"{__name__} ThreadTicker for {self.ticker}: run() method finished. Event loop exited."
        )
