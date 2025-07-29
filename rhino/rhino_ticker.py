"""
Ticker 毎のデータ処理クラス（銘柄スレッド・クラス）
機能スコープ
1. Realtime PSAR
"""
import logging

from PySide6.QtCore import (
    QObject,
    QThread,
    Signal,
    Slot,
)

from rhino.rhino_psar import PSARObject, RealtimePSAR


class TickerWorker(QObject):
    # Parabolic SAR の情報を通知
    notifyPSAR = Signal(str, float, PSARObject)

    def __init__(self, code: str, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.code = code
        self.psar = RealtimePSAR(code)

    @Slot(float, float)
    def addPrice(self, x, y):
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # Realtime PSAR の算出
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        ret: PSARObject = self.psar.add(y)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 Parabolic SAR の情報を通知
        self.notifyPSAR.emit(self.code, x, ret)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class Ticker(QThread):
    """
    各銘柄専用のスレッド
    """
    notifyNewPrice = Signal(float, float)

    # このスレッドが開始されたことを通知するシグナル（デバッグ用など）
    threadReady = Signal(str)

    def __init__(self, code: str, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.code = code
        self.worker = worker = TickerWorker(code)
        worker.moveToThread(self)  # TickerWorkerをこのQThreadに移動

        # スレッド開始時にworkerの準備完了を通知 (必要であれば)
        self.started.connect(self.thread_ready)

        # メインスレッドからワーカースレッドへ新たな株価情報を通知
        self.notifyNewPrice.connect(self.worker.addPrice)

    def thread_ready(self):
        self.threadReady.emit(self.code)

    def run(self):
        """
        このスレッドのイベントループを開始する。
        これがなければ、スレッドはすぐに終了してしまう。
        """
        self.logger.info(
            f"{__name__} ThreadTicker for {self.code}: run() method started. Entering event loop..."
        )
        self.exec()  # イベントループを開始
        self.logger.info(
            f"{__name__} ThreadTicker for {self.code}: run() method finished. Event loop exited."
        )
