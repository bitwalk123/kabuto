import logging

import pandas as pd
from PySide6.QtCore import QObject, Signal, Slot

from funcs.ios import load_excel
from funcs.tse import get_ticker_name_list
from modules.posman import PositionManager


class ExcelReviewWorker(QObject):
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
        self._running = True
        self.excel_path = excel_path
        self.dict_sheet = dict()

        # 銘柄リスト
        self.list_code = list()

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

        # 取得した Excel のシート名を銘柄コード (code) として扱う
        self.list_code = list(self.dict_sheet.keys())

        # 銘柄コードから銘柄名を取得
        dict_name = get_ticker_name_list(self.list_code)

        # デバッグ・モードでは、現在のところは前日終値を 0 とする
        dict_lastclose = dict()
        for code in self.list_code:
            dict_lastclose[code] = 0

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 銘柄名（リスト）などの情報を通知
        self.notifyTickerN.emit(
            self.list_code, dict_name, dict_lastclose
        )
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # ポジション・マネージャの初期化
        self.posman.initPosition(self.list_code)

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 データ読み込み済み（現時点では常に True を通知）
        self.notifyDataReady.emit(True)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    @Slot(float)
    def readCurrentPrice(self, ts: float):
        dict_data = dict()
        dict_profit = dict()
        dict_total = dict()
        for code in self.list_code:
            df = self.dict_sheet[code]
            # 指定された時刻から +1 秒未満で株価が存在するか確認
            df_tick = df[(ts <= df['Time']) & (df['Time'] < ts + 1)]
            if len(df_tick) > 0:
                # 時刻が存在していれば、データにある時刻と株価を返値に設定
                ts = df_tick.iloc[0, 0]
                price = df_tick.iloc[0, 1]
                dict_data[code] = [ts, price]
                dict_profit[code] = self.posman.getProfit(code, price)
                dict_total[code] = self.posman.getTotal(code)
            else:
                continue

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 現在時刻と株価、含み損、総収益を通知
        self.notifyCurrentPrice.emit(
            dict_data, dict_profit, dict_total
        )
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def stop(self):
        self._running = False

    @Slot()
    def stopProcess(self):
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 スレッドの正常終了を通知
        self.threadFinished.emit(True)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
