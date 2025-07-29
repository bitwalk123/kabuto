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

from rhino.rhino_funcs import get_default_psar_params
from rhino.rhino_psar import PSARObject, RealtimePSAR
from structs.res import AppRes


class TickerWorker(QObject):
    # Parabolic SAR の情報を通知
    notifyPSAR = Signal(str, float, PSARObject)
    # Parabolic SAR 関連パラメータを通知
    notifyPSARParams = Signal(dict)

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
        # 銘柄コード固有の設定ファイル名
        file_json = os.path.join(
            self.res.dir_conf,
            f"{self.code}.json"
        )

        if os.path.isfile(file_json):
            # 銘柄コード固有のファイルが存在すれば読み込む
            with open(file_json) as f:
                dict_psar = json.load(f)
        else:
            # デフォルトのパラメータ設定を取得
            dict_psar = get_default_psar_params()
            # 銘柄コード固有のファイルとして保存
            with open(file_json, "w") as f:
                json.dump(dict_psar, f)

        return dict_psar

    def getPSARParams(self):
        dict_psar = self.get_psar_params()
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 Parabolic SAR 関連のパラメータを通知
        self.notifyPSARParams.emit(dict_psar)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class Ticker(QThread):
    """
    各銘柄専用のスレッド
    """
    # 新たな株価情報をスレッドへ通知
    notifyNewPrice = Signal(float, float)
    # Parabolic SAR 関連のパラメータをリクエスト
    requestPSARParams = Signal()

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

        # 新たな株価情報を追加するメソッドへキューイング
        self.notifyNewPrice.connect(worker.addPrice)

        # Parabolic SAR 関連のパラメータを取得するメソッドへキューイング
        self.requestPSARParams.connect(worker.getPSARParams)

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
