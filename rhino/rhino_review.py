import logging

import pandas as pd
from PySide6.QtCore import (
    QObject,
    Signal,
    QThread, Slot,
)

from funcs.ios import load_excel
from funcs.tse import get_ticker_name_list
from modules.position_mannager import PositionManager
from structs.posman import PositionType


class RhinoReviewWorker(QObject):
    """
    Excel 形式の過去データを読み込むスレッドワーカー
    """
    # 銘柄名（リスト）通知シグナル
    notifyTickerN = Signal(list, dict, dict)

    # データ読み込み済み
    notifyDataReady = Signal(bool)

    # ティックデータ通知シグナル
    notifyCurrentPrice = Signal(dict, dict, dict)

    # 取引結果のデータフレーム通知シグナル
    notifyTransactionResult = Signal(pd.DataFrame)

    # スレッド終了シグナル（成否の論理値）
    threadFinished = Signal(bool)

    def __init__(self, excel_path: str):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.excel_path = excel_path
        self.dict_sheet = dict()

        # 銘柄リスト
        self.list_ticker = list()

        # ポジション・マネージャのインスタンス
        self.posman = PositionManager()

    @Slot()
    def getTransactionResult(self):
        """
        取引結果を取得
        :return:
        """
        df = self.posman.getTransactionResult()
        self.notifyTransactionResult.emit(df)

    @Slot()
    def loadExcel(self):
        """
        ティックデータを保存した Excel ファイルの読み込み
        :return:
        """
        try:
            self.dict_sheet = load_excel(self.excel_path)
        except Exception as e:
            msg = "encountered error in reading Excel file:"
            self.logger.critical(f"{__name__}: {msg} {e}")
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # 🧿 スレッドの異常終了を通知
            self.threadFinished.emit(False)
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            return

        # 取得した Excel のシート名を銘柄コード (ticker) として扱う
        self.list_ticker = list(self.dict_sheet.keys())

        # 銘柄コードから銘柄名を取得
        dict_name = get_ticker_name_list(self.list_ticker)

        # デバッグ・モードでは、現在のところは前日終値を 0 とする
        dict_lastclose = dict()
        for ticker in self.list_ticker:
            dict_lastclose[ticker] = 0

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 銘柄名（リスト）などの情報を通知
        self.notifyTickerN.emit(
            self.list_ticker, dict_name, dict_lastclose
        )
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # ポジション・マネージャの初期化
        self.posman.initPosition(self.list_ticker)

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 データ読み込み済み（現時点では常に True を通知）
        self.notifyDataReady.emit(True)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    @Slot(float)
    def readCurrentPrice(self, ts: float):
        dict_data = dict()
        dict_profit = dict()
        dict_total = dict()
        for ticker in self.list_ticker:
            df = self.dict_sheet[ticker]
            # 指定された時刻から +1 秒未満で株価が存在するか確認
            df_tick = df[(ts <= df['Time']) & (df['Time'] < ts + 1)]
            if len(df_tick) > 0:
                # 時刻が存在していれば、データにある時刻と株価を返値に設定
                ts = df_tick.iloc[0, 0]
                price = df_tick.iloc[0, 1]
                dict_data[ticker] = [ts, price]
                dict_profit[ticker] = self.posman.getProfit(ticker, price)
                dict_total[ticker] = self.posman.getTotal(ticker)
            else:
                continue

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 現在時刻と株価、含み損、総収益を通知
        self.notifyCurrentPrice.emit(
            dict_data, dict_profit, dict_total
        )
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    @Slot()
    def stopProcess(self):
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 スレッドの正常終了を通知
        self.threadFinished.emit(True)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class RhinoReview(QThread):
    # ワーカーの初期化シグナル
    requestWorkerInit = Signal()

    # 現在価格取得リクエスト・シグナル
    requestCurrentPrice = Signal(float)

    # 売買シグナル
    requestPositionOpen = Signal(str, float, float, PositionType, str)
    requestPositionClose = Signal(str, float, float, str)
    requestTransactionResult = Signal()

    def __init__(self, excel_path: str):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        # データ (Excel) 読み込み済みかどうかのフラグ
        self.flag_data_ready = False

        # ワーカースレッド・インスタンスの生成およびスレッドへの移動
        self.worker = worker = RhinoReviewWorker(excel_path)
        worker.notifyDataReady.connect(self.set_data_ready_status)
        worker.threadFinished.connect(self.quit)  # スレッド終了時
        worker.moveToThread(self)

        # ---------------------------------------------------------------------
        # スレッドが開始されたら、ワーカースレッド内で初期化処理を実行するシグナルを発行
        self.started.connect(self.requestWorkerInit.emit)
        # ---------------------------------------------------------------------
        # 初期化処理は指定された Excel ファイルの読み込み
        self.requestWorkerInit.connect(worker.loadExcel)
        # ---------------------------------------------------------------------
        # 売買ポジション処理用のメソッドへキューイング
        self.requestPositionOpen.connect(worker.posman.openPosition)
        self.requestPositionClose.connect(worker.posman.closePosition)
        # ---------------------------------------------------------------------
        # 取引結果を取得するメソッドへキューイング
        self.requestTransactionResult.connect(worker.getTransactionResult)
        # ---------------------------------------------------------------------
        # 現在株価を取得するメソッドへキューイング。
        self.requestCurrentPrice.connect(worker.readCurrentPrice)
        # ---------------------------------------------------------------------
        # スレッド終了関連
        self.finished.connect(self.deleteLater)  # スレッドオブジェクトの削除

    def isDataReady(self) -> bool:
        return self.flag_data_ready

    def run(self):
        """
        このスレッドのイベントループを開始する。
        これがなければ、スレッドはすぐに終了してしまう。
        """
        self.logger.info(
            f"{__name__}: run() method started. Entering event loop..."
        )
        self.exec()  # イベントループを開始
        self.logger.info(
            f"{__name__}: run() method finished. Event loop exited."
        )

    def set_data_ready_status(self, state: bool):
        self.flag_data_ready = state
        self.logger.info(
            f"{__name__}: now, data ready flag becomes {state}!"
        )
