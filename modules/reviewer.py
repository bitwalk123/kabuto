import logging
import time

import pandas as pd
from PySide6.QtCore import QObject, Signal, Slot

from funcs.excel import load_excel
from funcs.tse import get_ticker_name_list
from modules.posman import PositionManager
from structs.app_enum import ActionType


class ExcelReviewWorker(QObject):
    """
    Excel 形式の過去データを読み込むスレッドワーカー
    """
    # 1. 銘柄名（リスト）通知シグナル
    notifyTickerN = Signal(list, dict)
    # 2. ティックデータを通知
    notifyCurrentPrice = Signal(dict, dict, dict)
    # 3. 取引結果のデータフレームを通知
    notifyTransactionResult = Signal(pd.DataFrame)
    # 4, 約定確認結果を通知
    sendResult = Signal(str, float)
    # 5. ティックデータ保存の終了を通知（本番のみ - デバッグ用ではダミー）
    saveCompleted = Signal(bool)
    # 6. データ準備完了（デバッグ用）
    notifyDataReady = Signal(bool)
    # 7. スレッド終了シグナル（成否の論理値）
    threadFinished = Signal(bool)

    def __init__(self, excel_path: str) -> None:
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.prefix = self.__class__.__name__
        self._running: bool = True
        self.excel_path = excel_path
        self.dict_sheet: dict[str, pd.DataFrame] = {}

        # 銘柄リスト
        self.list_code: list[str] = []

        # ポジション・マネージャのインスタンス
        self.posman: PositionManager = PositionManager()

    @Slot()
    def getTransactionResult(self) -> None:
        """
        取引結果を取得
        """
        df = self.posman.getTransactionResult()
        self.notifyTransactionResult.emit(df)

    @Slot()
    def initWorker(self) -> None:
        """
        ティックデータを保存した Excel ファイルの読み込み
        """
        try:
            self.dict_sheet = load_excel(self.excel_path)
        except Exception as e:
            msg = f"encountered error in reading Excel, {self.excel_path}:"
            self.logger.critical(f"{msg} {e}")
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # 🧿 スレッドの異常終了を通知
            self.threadFinished.emit(False)
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            return

        # 取得した Excel のシート名を銘柄コード (code) として扱う
        self.list_code = list(self.dict_sheet.keys())

        # 銘柄コードから銘柄名を取得
        dict_name: dict[str, str] = get_ticker_name_list(self.list_code)

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 銘柄名（リスト）などの情報を通知
        self.notifyTickerN.emit(
            self.list_code, dict_name
        )
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # ポジション・マネージャの初期化
        self.posman.initPosition(self.list_code)

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 データ読み込み済み（現時点では常に True を通知）
        self.notifyDataReady.emit(True)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    @Slot(float)
    def readCurrentPrice(self, t: float) -> None:
        dict_data: dict[str, tuple[float, float, float]] = {}
        dict_profit: dict[str, float] = {}
        dict_total: dict[str, float] = {}

        for code in self.list_code:
            df: pd.DataFrame = self.dict_sheet[code]
            # 指定された時刻から +1 秒未満で株価が存在するか確認
            df_tick: pd.DataFrame = df[(t <= df["Time"]) & (df["Time"] < t + 1)]  # type: ignore
            if len(df_tick) > 0:
                # 時刻が存在していれば、データにある時刻と株価を返値に設定
                row = df_tick.iloc[0]
                ts = row["Time"]
                price = row["Price"]
                volume = row["Volume"]
                # メイン・スレッドへ渡す情報を準備
                dict_data[code] = (ts, price, volume)
                dict_profit[code] = self.posman.getProfit(code, price)
                dict_total[code] = self.posman.getTotal(code)

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 現在時刻と株価、含み損、総収益を通知
        self.notifyCurrentPrice.emit(
            dict_data, dict_profit, dict_total
        )
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def saveDataFrame(self) -> None:
        """
        デバッグ用ではダミー
        """
        pass

    def stop(self) -> None:
        self._running = False

    @Slot()
    def stopProcess(self) -> None:
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 スレッドの正常終了を通知
        self.threadFinished.emit(True)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
    # 取引ボタンがクリックされた時の処理
    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
    @Slot(str, float, float, str)
    def macro_do_buy(self, code: str, ts: float, price: float, note: str) -> None:
        time.sleep(0.2)
        # 買建で新規建玉
        self.posman.openPosition(code, ts, price, ActionType.BUY, note)
        self.sendResult.emit(code, price)

    @Slot(str, float, float, str)
    def macro_do_sell(self, code: str, ts: float, price: float, note: str) -> None:
        time.sleep(0.2)
        # 売建で新規建玉
        self.posman.openPosition(code, ts, price, ActionType.SELL, note)
        self.sendResult.emit(code, price)

    @Slot(str, float, float, str)
    def macro_do_repay(self, code: str, ts: float, price: float, note: str) -> None:
        time.sleep(0.2)
        # 建玉返済
        self.posman.closePosition(code, ts, price, note)
        self.sendResult.emit(code, price)
