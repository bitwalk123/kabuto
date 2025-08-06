"""
Ticker 毎のデータ処理クラス（銘柄スレッド・クラス）
機能スコープ
1. Realtime PSAR
2. Trend Chaser
3. Smoothing Spline
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
from structs.app_enum import FollowType
from structs.res import AppRes


class TickerWorker(QObject):
    # Parabolic SAR 関連のデフォルトのパラメータを通知
    notifyDefaultPSARParams = Signal(dict)
    # Parabolic SAR の情報を通知
    notifyPSAR = Signal(str, float, PSARObject)
    # Parabolic SAR 関連パラメータを通知
    notifyPSARParams = Signal(dict)
    # Over Drive の状態変更の通知
    notifyODStatusChanged = Signal(bool)

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
        if ret.follow == FollowType.OVERDRIVE and ret.overdrive is False:
            ret.overdrive = True
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # 🧿 Over Drive の状態変更の通知
            self.notifyODStatusChanged.emit(True)
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        elif ret.follow == FollowType.PARABOLIC and ret.overdrive is True:
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # 🧿 Over Drive の状態変更の通知
            ret.overdrive = False
            self.notifyODStatusChanged.emit(False)
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    @staticmethod
    def get_default_psar_params() -> dict:
        """
        デフォルトの Parabolic SAR 関連のパラメータを返す関数
        :return:
        """
        dict_psar = dict()

        # for Parabolic SAR
        dict_psar["af_init"]: float = 0.000005
        dict_psar["af_step"]: float = 0.000005
        dict_psar["af_max"]: float = 0.005
        # for Trend Chaser
        dict_psar["factor_d"]: float = 10  # 許容される ys と PSAR の最大差異
        dict_psar["factor_c"]: float = 0.95  # ys と psar の間を縮める係数

        # for Smoothing Spline
        dict_psar["power_lam"]: int = 6  # Lambda for smoothing spline
        dict_psar["n_smooth_min"]: int = 150  # dead time (min) at start up
        dict_psar["n_smooth_max"]: int = 600  # maximum data for smoothing

        return dict_psar

    def get_json_path(self) -> str:
        """
        銘柄コードに対応した JSON ファイルのパスを取得
        :return:
        """
        file_json = os.path.join(self.res.dir_conf, f"{self.code}.json")
        return file_json

    def get_psar_params(self) -> dict:
        # 銘柄コード固有の設定ファイル名
        file_json = self.get_json_path()

        if os.path.isfile(file_json):
            # 銘柄コード固有のファイルが存在すれば読み込む
            dict_psar = self.read_contents_from_json(file_json)
        else:
            # デフォルトのパラメータ設定を取得
            dict_psar = self.get_default_psar_params()
            # 銘柄コード固有のファイルとして保存
            self.save_contents_to_json(file_json, dict_psar)

        return dict_psar

    def getDefaultPSARParams(self):
        # デフォルトのパラメータ設定を取得
        dict_psar = self.get_default_psar_params()
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 Parabolic SAR 関連のパラメータを通知
        self.notifyDefaultPSARParams.emit(dict_psar)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def getPSARParams(self):
        """
        パラメータ設定の取得要求に対する応答
        :return:
        """
        dict_psar = self.get_psar_params()
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 Parabolic SAR 関連のパラメータを通知
        self.notifyPSARParams.emit(dict_psar)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def changeOverDriveStatus(self, state: bool):
        """
        Over Drive 状態の変更
        :param state:
        :return:
        """
        self.psar.setOverDriveStatus(state)

    def updatePSARParams(self, dict_psar):
        """
        パラメータ設定の更新要求に対する応答（付与された辞書を保存）
        :param dict_psar:
        :return:
        """
        # 新しいパラメータを該当する銘柄コードの JSON へ保存
        file_json = self.get_json_path()
        self.save_contents_to_json(file_json, dict_psar)
        self.logger.info(f"{__name__}: {file_json}'s been updated.")
        # 新しいパラメータを現 PSAR オブジェクトへ反映
        self.psar.setPSARParams(dict_psar)
        self.logger.info(f"{__name__}: new params's been set to the PSAR object.")

    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
    #  JSON 入出力関連
    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
    @staticmethod
    def read_contents_from_json(file_json) -> dict:
        with open(file_json) as f:
            dict_psar = json.load(f)

        return dict_psar

    @staticmethod
    def save_contents_to_json(file_json: str, dict_psar: dict):
        with open(file_json, "w") as f:
            json.dump(dict_psar, f)


class Ticker(QThread):
    """
    各銘柄専用のスレッド
    """
    # 新たな株価情報をスレッドへ通知
    notifyNewPrice = Signal(float, float)
    # Parabolic SAR 関連のデフォルトのパラメータをリクエスト
    requestDefaultPSARParams = Signal()
    # Parabolic SAR 関連のパラメータをリクエスト
    requestPSARParams = Signal()
    # Parabolic SAR 関連のパラメータを更新
    requestUpdatePSARParams = Signal(dict)
    # Over Drive の状態変更の要求
    requestOEStatusChange = Signal(bool)

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
        self.requestDefaultPSARParams.connect(worker.getDefaultPSARParams)

        # Parabolic SAR 関連のパラメータを取得するメソッドへキューイング
        self.requestPSARParams.connect(worker.getPSARParams)

        # Parabolic SAR 関連のパラメータを更新するメソッドへキューイング
        self.requestUpdatePSARParams.connect(worker.updatePSARParams)

        # Over Drive 状態の変更
        self.requestOEStatusChange.connect(worker.changeOverDriveStatus)

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
