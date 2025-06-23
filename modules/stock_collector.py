import logging
import sys

import pandas as pd
import xlwings as xw

# Windows 固有のライブラリ
if sys.platform == "win32":
    from pywintypes import com_error

from PySide6.QtCore import QObject, QThread, Signal

from structs.res import AppRes


class StockCollectorWorker(QObject):
    # 銘柄名（リスト）の通知
    notifyTickerN = Signal(list, dict)

    def __init__(self, excel_path: str):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.excel_path = excel_path

        # ---------------------------------------------------------------------
        # xlwings のインスタンス
        # この初期化プロセスでは xlwings インスタンスの初期化ができない。
        # Excel と通信する COM オブジェクトがスレッドアフィニティ（特定のCOMオブジェクトは
        # 特定のシングルスレッドアパートメントでしか動作できないという制約）を持っているため
        # ---------------------------------------------------------------------
        self.wb = None  # Excel のワークブックインスタンス
        self.sheet = None  # Excel のワークシートインスタンス

        # Excelシートから xlwings でデータを読み込むときの試行回数
        # 楽天証券のマーケットスピード２ RSS の書込と重なる（衝突する）と、
        # COM エラーが発生するため、リトライできるようにしている。
        self.max_retries = 3  # 最大リトライ回数
        self.retry_delay = 0.1  # リトライ間の遅延（秒）
        # ---------------------------------------------------------------------

        # Excel ワークシート情報
        self.cell_bottom = "------"
        self.list_ticker = list()  # 銘柄リスト
        self.dict_row = dict()  # 銘柄の行位置
        self.dict_name = dict()  # 銘柄名
        self.dict_df = dict()  # 銘柄別データフレーム

        # Excel の列情報
        self.col_code = 0  # 銘柄コード
        self.col_name = 1  # 銘柄名
        self.col_date = 2  # 日付
        self.col_time = 3  # 時刻
        self.col_price = 4  # 現在詳細株価
        self.col_lastclose = 5  # 前日終値

    def initWorker(self):
        #######################################################################
        # 情報を取得する Excel ワークブック・インスタンスの生成
        self.wb = wb = xw.Book(self.excel_path)
        name_sheet = "Cover"
        self.sheet = wb.sheets[name_sheet]
        #
        #######################################################################

        row = 1
        flag_loop = True
        while flag_loop:
            ticker = self.sheet[row, self.col_code].value
            if ticker == self.cell_bottom:
                flag_loop = False
            else:
                # 銘柄コード
                self.list_ticker.append(ticker)

                # 行位置
                self.dict_row[ticker] = row

                # 銘柄名
                self.dict_name[ticker] = self.sheet[row, self.col_name].value

                # 銘柄別に空のデータフレームを準備
                self.dict_df[ticker] = pd.DataFrame({
                    "Time": list(),
                    "Price": list()
                })

                # 行番号のインクリメント
                row += 1

        # --------------------------------------------------------------
        # 🧿 銘柄名などの情報を通知
        self.notifyTickerN.emit(self.list_ticker, self.dict_name)
        # --------------------------------------------------------------


class StockCollector(QThread):
    requestWorkerInit = Signal()

    # このスレッドが開始されたことを通知するシグナル（デバッグ用など）
    threadReady = Signal()

    def __init__(self, res: AppRes):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.res = res

        excel_path = res.excel_collector
        self.worker = worker = StockCollectorWorker(excel_path)
        self.worker.moveToThread(self)  # ThreadStockCollectorWorkerをこのQThreadに移動
        # QThread が開始されたら、ワーカースレッド内で初期化処理を開始するシグナルを発行
        self.started.connect(self.requestWorkerInit.emit)

        # スレッド開始時にworkerの準備完了を通知 (必要であれば)
        self.started.connect(self.thread_ready)

        # _____________________________________________________________________
        # メイン・スレッド側のシグナルとワーカー・スレッド側のスロット（メソッド）の接続
        # 初期化処理は指定された Excel ファイルを読み込むこと
        # xlwings インスタンスを生成、Excel の銘柄情報を読込むメソッドへキューイング。
        self.requestWorkerInit.connect(worker.initWorker)

    def thread_ready(self):
        self.threadReady.emit()

    def run(self):
        """
        このスレッドのイベントループを開始する。
        これがなければ、スレッドはすぐに終了してしまう。
        """
        self.logger.info(
            f"{__name__} StockCollector: run() method started. Entering event loop..."
        )
        self.exec()  # イベントループを開始
        self.logger.info(
            f"{__name__} StockCollector: run() method finished. Event loop exited."
        )
