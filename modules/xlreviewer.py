import datetime
import logging

from PySide6.QtCore import QObject, Signal

from funcs.ios import load_excel


class ReviewWorker(QObject):
    # 銘柄名（リスト）の通知
    notifyTickerN = Signal(list, dict)

    # スレッド完了シグナル（成否の論理値）
    threadFinished = Signal(bool)

    def __init__(self, excel_path: str):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.excel_path = excel_path

    def loadExcel(self):
        try:
            dict_sheet = load_excel(self.excel_path)
        except Exception as e:
            msg = "Excelファイルの読み込み中にエラーが発生しました:"
            self.logger.critical(f"{msg} {e}")
            self.threadFinished.emit(False)
            return

        dict_times = dict()

        for ticker in dict_sheet.keys():
            df = dict_sheet[ticker]
            dt = datetime.datetime.fromtimestamp(df['Time'].iloc[0])
            dt_start = datetime.datetime(dt.year, dt.month, dt.day, hour=9, minute=0)
            dt_end = datetime.datetime(dt.year, dt.month, dt.day, hour=15, minute=30)
            dict_times[ticker] = [dt_start.timestamp(), dt_end.timestamp()]

        # 銘柄名（リスト）の通知
        self.notifyTickerN.emit(list(dict_sheet.keys()), dict_times)

        # スレッドの終了
        #self.threadFinished.emit(True)
