"""
Ticker 毎のデータ処理クラス（銘柄スレッド・クラス）
機能スコープ
1. Realtime PSAR
"""
import json
import logging
import os

from PySide6.QtCore import (
    QObject,
    QThread,
    Signal,
    Slot,
)

from rhino.rhino_psar import PSARObject, RealtimePSAR
from structs.res import AppRes


class TickerWorker(QObject):
    # Parabolic SAR の情報を通知
    notifyPSAR = Signal(str, float, PSARObject)

    def __init__(self, res: AppRes, code: str, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.res = res
        self.code = code

        dict_psar = self.get_psar_params()
        self.psar = RealtimePSAR(dict_psar)

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

    def get_psar_params(self) -> dict:
        # 銘柄コード固有の設定ファイル
        file_json = os.path.join(
            self.res.dir_conf,
            f"{self.code}.json"
        )

        if os.path.isfile(file_json):
            # 銘柄コード固有のファイルが存在すれば読み込む
            with open(file_json) as f:
                dict_psar = json.load(f)
        else:
            dict_psar = dict()
            # for Parabolic SAR
            dict_psar["af_init"]: float = 0.000005
            dict_psar["af_step"]: float = 0.000005
            dict_psar["af_max"]: float = 0.005
            dict_psar["factor_d"] = 20  # 許容される ys と PSAR の最大差異
            # for smoothing
            dict_psar["power_lam"]: int = 7
            dict_psar["n_smooth_min"] = 60
            dict_psar["n_smooth_max"] = 600
            # 銘柄コード固有のファイルとして保存
            with open(file_json, "w") as f:
                json.dump(dict_psar, f)

        return dict_psar


class Ticker(QThread):
    """
    各銘柄専用のスレッド
    """
    notifyNewPrice = Signal(float, float)

    # このスレッドが開始されたことを通知するシグナル（デバッグ用など）
    threadReady = Signal(str)

    def __init__(self, res: AppRes, code: str, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.code = code
        self.worker = worker = TickerWorker(res, code)
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
