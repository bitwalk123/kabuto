import datetime
import logging

from PySide6.QtCore import QObject, Signal

from funcs.ios import load_excel


class ReviewWorker(QObject):
    """
    Excel 形式の過去データを読み込むスレッドワーカー
    """
    # 銘柄名（リスト）の通知
    notifyTickerN = Signal(list, dict)

    # ティックデータの表示
    notifyCurrentPrice = Signal(dict)

    # スレッド終了シグナル（成否の論理値）
    threadFinished = Signal(bool)

    def __init__(self, excel_path: str):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.excel_path = excel_path
        self.dict_sheet = dict()

    def loadExcel(self):
        try:
            self.dict_sheet = load_excel(self.excel_path)
        except Exception as e:
            msg = "Excelファイルの読み込み中にエラーが発生しました:"
            self.logger.critical(f"{msg} {e}")
            self.threadFinished.emit(False)
            return

        dict_times = dict()

        for ticker in self.dict_sheet.keys():
            df = self.dict_sheet[ticker]
            dt = datetime.datetime.fromtimestamp(df['Time'].iloc[0])
            dt_start = datetime.datetime(dt.year, dt.month, dt.day, hour=9, minute=0)
            dt_end = datetime.datetime(dt.year, dt.month, dt.day, hour=15, minute=25)
            dict_times[ticker] = [dt_start.timestamp(), dt_end.timestamp()]

        # 銘柄名（リスト）の通知
        self.notifyTickerN.emit(list(self.dict_sheet.keys()), dict_times)

        # スレッドの終了
        # self.threadFinished.emit(True)

    def readCurrentPrice(self, ts: float):
        dict_data = dict()
        for ticker in self.dict_sheet.keys():
            df = self.dict_sheet[ticker]
            df_tick = df[(ts <= df['Time']) & (df['Time'] < ts + 1)]
            if len(df_tick) > 0:
                time = df_tick.iloc[0, 0]
                price = df_tick.iloc[0, 1]
                dict_data[ticker] = [time, price]
            else:
                dict_data[ticker] = [ts, 0]

        self.notifyCurrentPrice.emit(dict_data)