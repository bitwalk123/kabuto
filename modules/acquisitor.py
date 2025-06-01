import logging
import sys
import time

import xlwings as xw

if sys.platform == "win32":
    from pywintypes import com_error  # Windows 固有のライブラリ

from PySide6.QtCore import QObject, Signal


class AquireWorker(QObject):
    notifyTickerN = Signal(list, dict, dict)

    # スレッド終了シグナル（成否の論理値）
    threadFinished = Signal(bool)

    def __init__(self, excel_path: str):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.excel_path = excel_path

        self.wb = None
        self.sheet = None

        # Excelシートから xlwings でデータを読み込むときの試行回数
        self.max_retries = 3  # 最大リトライ回数
        self.retry_delay = 0.1  # リトライ間の遅延（秒）

        # 銘柄リスト
        self.list_ticker = list()

        # 列情報
        self.col_code = 0
        self.col_name = 1
        self.col_date = 2
        self.col_time = 3
        self.col_price = 4
        self.col_lastclose = 5

        # 最大銘柄数
        self.num_max = 3

    def loadExcel(self):
        #######################################################################
        # 情報を取得する Excel ファイル
        self.wb = wb = xw.Book(self.excel_path)
        self.sheet = wb.sheets["Sheet1"]

        dict_name = dict()
        dict_lastclose = dict()
        for num in range(self.num_max):
            row = num + 1

            # 銘柄コード
            ticker = self.sheet[row, self.col_code].value
            self.list_ticker.append(ticker)

            # 銘柄名
            dict_name[ticker] = self.sheet[row, self.col_name].value

            # 前日の終値の横線
            dict_lastclose[ticker] = self.sheet[row, self.col_lastclose].value
        #
        #######################################################################
        self.threadFinished.emit(self.list_ticker, dict_name, dict_lastclose)

    def read_price(self):
        for i, ticker in enumerate(self.list_ticker):
            row = i + 1
            # Excel シートから株価情報を取得
            for attempt in range(self.max_retries):
                # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
                # 楽天証券のマーケットスピード２ RSS の書き込みと重なる（衝突する）と、
                # COM エラーが発生するためリトライできるようにしている。
                # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
                try:
                    # Excelシートから株価データを取得
                    y = self.sheet[row, self.col_price].value
                    break
                except com_error as e:
                    # com_error は Windows 固有
                    if attempt < self.max_retries - 1:
                        self.logger.warning(
                            f"{__name__} COM error occurred, retrying... (Attempt {attempt + 1}/{self.max_retries}) Error: {e}"
                        )
                        time.sleep(self.retry_delay)
                    else:
                        self.logger.error(
                            f"{__name__} COM error occurred after {self.max_retries} attempts. Giving up."
                        )
                        raise  # 最終的に失敗したら例外を再発生させる
                except Exception as e:
                    self.logger.exception(f"{__name__} an unexpected error occurred: {e}")
                    raise  # その他の例外はそのまま発生させる
            print(ticker, y)

    def stop_processing(self):
        # self.running = False
        if self.wb:
            try:
                self.wb.close()  # ブックを閉じる
                print("Worker: Excel book closed.")
            except Exception as e:
                print(f"Worker: Error closing book: {e}")
            # ブックを閉じた後、その親アプリケーションも終了させる
            if self.wb.app:
                try:
                    self.wb.app.quit()
                    print("Worker: Excel app quit.")
                except Exception as e:
                    print(f"Worker: Error quitting app: {e}")
            self.book = None  # オブジェクト参照をクリア
        self.threadFinished.emit()
