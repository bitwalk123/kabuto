"""
Ticker 毎のデータ処理クラス（銘柄スレッド・クラス）
機能スコープ
1. Realtime PSAR
2. Moving Range
"""
import logging
from collections import deque

from PySide6.QtCore import (
    QObject,
    QThread,
    Signal,
    Slot,
)

from modules.psar import RealtimePSAR


class TickerWorker(QObject):
    # Parabolic SAR の情報を通知
    notifyPSAR = Signal(str, int, float, float)
    notifyMR = Signal(str, float, float)

    def __init__(self, ticker, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.ticker = ticker
        self.psar = RealtimePSAR()
        self.period = 30
        self.deque_mr = deque(maxlen=self.period)

    @Slot(float, float)
    def addPrice4PSAR(self, x, y):
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # Realtime PSAR の算出
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        ret = self.psar.add(y)
        # トレンドと PSAR の値を転記
        trend = ret.trend
        y_psar = ret.psar
        # --------------------------------------------------------
        # 🧿 Parabolic SAR の情報を通知
        self.notifyPSAR.emit(self.ticker, trend, x, y_psar)
        # --------------------------------------------------------

        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # Moving Ranga の算出
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        self.deque_mr.append(y)
        y_mr = max(self.deque_mr) - min(self.deque_mr)
        # ---------------------------------------------
        # 🧿 MR の情報を通知
        self.notifyMR.emit(self.ticker, x, y_mr)
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

    # デストラクタでクリーンアップが必要な場合はここに記述
    # def __del__(self):
    #     print(f"ThreadTicker {self.ticker_code} is being deleted.")
