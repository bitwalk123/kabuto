import logging

import pandas as pd
from PySide6.QtCore import QObject, Signal

from funcs.ios import load_excel
from funcs.tse import get_ticker_name_list
from modules.position_mannager import PositionManager
from modules.psar import RealtimePSAR


class ReviewWorker(QObject):
    """
    Excel 形式の過去データを読み込むスレッドワーカー
    """
    # 銘柄名（リスト）の通知
    notifyTickerN = Signal(list, dict, dict)

    # ティックデータを通知
    notifyCurrentPrice = Signal(dict, dict, dict)

    # 取引結果のデータフレームを通知
    notifyTransactionResult = Signal(pd.DataFrame)

    # Parabolic SAR の情報を通知
    notifyPSAR = Signal(str, int, float, float)

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

        # Parabolic SAR の辞書
        self.dict_psar = dict()

    def getTransactionResult(self):
        """
        取引結果を取得
        :return:
        """
        df = self.posman.getTransactionResult()
        self.notifyTransactionResult.emit(df)

    def loadExcel(self):
        """
        ティックデータを保存した Excel ファイルの読み込み
        :return:
        """
        try:
            self.dict_sheet = load_excel(self.excel_path)
        except Exception as e:
            msg = "Excelファイルの読み込み中にエラーが発生しました:"
            self.logger.critical(f"{msg} {e}")
            # ------------------------------
            # 🧿 スレッドの異常終了を通知
            self.threadFinished.emit(False)
            # ------------------------------
            return

        # 取得した Excel のシート名を銘柄コード (ticker) として扱う
        self.list_ticker = list(self.dict_sheet.keys())
        # 銘柄コードから銘柄名を取得
        dict_name = get_ticker_name_list(self.list_ticker)
        # デバッグ・モードでは、現在のところは前日終値を 0 とする
        dict_lastclose = dict()
        for ticker in self.list_ticker:
            dict_lastclose[ticker] = 0
        # -----------------------------------------------
        # 🧿 銘柄名（リスト）などの情報を通知
        self.notifyTickerN.emit(
            self.list_ticker, dict_name, dict_lastclose
        )
        # -----------------------------------------------

        # ポジション・マネージャの初期化
        self.posman.initPosition(self.list_ticker)

        # Parabolic SAR インスタンスの生成
        for ticker in self.list_ticker:
            self.dict_psar[ticker] = RealtimePSAR()

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
                time = df_tick.iloc[0, 0]
                price = df_tick.iloc[0, 1]
                dict_data[ticker] = [time, price]
            else:
                # 存在しなければ、指定時刻と株価 = 0 を設定
                #price = 0
                #dict_data[ticker] = [ts, price]
                # 存在しなければ処理しない
                continue

            dict_profit[ticker] = self.posman.getProfit(ticker, price)
            dict_total[ticker] = self.posman.getTotal(ticker)

        # --------------------------------------
        # 🧿 現在時刻と株価を通知
        self.notifyCurrentPrice.emit(dict_data, dict_profit, dict_total)
        # --------------------------------------

        # Parabolic SAR の算出
        for ticker in dict_data.keys():
            x, y = dict_data[ticker]
            # ticker 毎に RealtimePSAR オブジェクトを取り出す
            psar: RealtimePSAR = self.dict_psar[ticker]
            # Realtime PSAR の算出
            ret = psar.add(y)
            # トレンドと PSAR の値を転記
            trend = ret.trend
            y_psar = ret.psar
            # ---------------------------------------------------
            # 🧿 Parabolic SAR の情報を通知
            self.notifyPSAR.emit(ticker, trend, x, y_psar)
            # ---------------------------------------------------

    def stopProcess(self):
        # -----------------------------
        # 🧿 スレッドの正常終了を通知
        self.threadFinished.emit(True)
        # -----------------------------
