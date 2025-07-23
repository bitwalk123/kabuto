import logging
import os
import sys
import time

import pandas as pd
import xlwings as xw

from funcs.ios import save_dataframe_to_excel
from funcs.tide import get_date_str_today

# Windows 固有のライブラリ
if sys.platform == "win32":
    from pywintypes import com_error

from PySide6.QtCore import QObject, QThread, Signal

from structs.res import AppRes


class RssConnectWorker(QObject):
    # 銘柄名（リスト）の通知
    notifyTickerList = Signal(list, dict)

    # ティックデータを通知
    notifyCurrentPrice = Signal(dict)

    # Excel 関数の実行結果を通知
    notifyExcelFuncResult = Signal(bool)

    # スレッドの終了を通知
    threadFinished = Signal()

    def __init__(self, res: AppRes, excel_path: str):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.res = res
        self.excel_path = excel_path
        self.order_no = 1

        # ---------------------------------------------------------------------
        # xlwings のインスタンス
        # この初期化プロセスでは xlwings インスタンスの初期化ができない。
        # Excel と通信する COM オブジェクトがスレッドアフィニティ（特定のCOMオブジェクトは
        # 特定のシングルスレッドアパートメントでしか動作できないという制約）を持っているため
        # ---------------------------------------------------------------------
        self.wb = None  # Excel のワークブックインスタンス
        self.sheet = None  # Excel のワークシートインスタンス
        # Excel 側の関数
        self.exec_buy = None
        self.exec_sell = None
        self.exec_repay = None

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
        self.logger.info(f"{self.__class__}: in init process.")
        #######################################################################
        # 情報を取得する Excel ワークブック・インスタンスの生成
        self.wb = wb = xw.Book(self.excel_path)
        name_sheet = "Cover"
        self.sheet = wb.sheets[name_sheet]

        self.exec_buy = wb.macro("ExecBuy")

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

                # 行番号のインクリメント
                row += 1

        # ---------------------------------------------------------------------
        # 🧿 銘柄名などの情報を通知
        self.notifyTickerList.emit(self.list_ticker, self.dict_name)
        # ---------------------------------------------------------------------

    def executeBuy(self, code: str):
        result = False
        for attempt in range(self.max_retries):
            try:
                # Excel の関数 ExecBuy の実行
                result = self.exec_buy(self.order_no, code)
                self.order_no += 1
                break
            except com_error as e:
                # ---------------------------------------------------------
                # com_error は Windows 固有
                # ---------------------------------------------------------
                if attempt < self.max_retries - 1:
                    self.logger.warning(
                        f"{self.__class__} COM error occurred, retrying... (Attempt {attempt + 1}/{self.max_retries}) Error: {e}"
                    )
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error(
                        f"{self.__class__} COM error occurred after {self.max_retries} attempts. Giving up."
                    )
                    raise  # 最終的に失敗したら例外を再発生させる
            except Exception as e:
                self.logger.exception(f"{self.__class__} an unexpected error occurred: {e}")
                raise  # その他の例外はそのまま発生させる

        # ---------------------------------------------------------------------
        # 🧿 銘柄名などの情報を通知
        self.notifyExcelFuncResult.emit(result)
        # ---------------------------------------------------------------------

    def readCurrentPrice(self):
        dict_data = dict()
        for ticker in self.list_ticker:
            row_excel = self.dict_row[ticker]
            # Excel シートから株価情報を取得
            for attempt in range(self.max_retries):
                ###############################################################
                # 楽天証券のマーケットスピード２ RSS の書込と重なる（衝突する）と、
                # COM エラーが発生するため、リトライできるようにしている。
                # -------------------------------------------------------------
                try:
                    ts = time.time()
                    # Excelシートから株価データを取得
                    price = self.sheet[row_excel, self.col_price].value
                    if price > 0:
                        # ここではタイムスタンプを時刻に採用する
                        dict_data[ticker] = [ts, price]
                    break
                except com_error as e:
                    # ---------------------------------------------------------
                    # com_error は Windows 固有
                    # ---------------------------------------------------------
                    if attempt < self.max_retries - 1:
                        self.logger.warning(
                            f"{self.__class__} COM error occurred, retrying... (Attempt {attempt + 1}/{self.max_retries}) Error: {e}"
                        )
                        time.sleep(self.retry_delay)
                    else:
                        self.logger.error(
                            f"{self.__class__} COM error occurred after {self.max_retries} attempts. Giving up."
                        )
                        raise  # 最終的に失敗したら例外を再発生させる
                except Exception as e:
                    self.logger.exception(f"{self.__class__} an unexpected error occurred: {e}")
                    raise  # その他の例外はそのまま発生させる
                #
                ###############################################################

        # ---------------------------------------------------------------------
        # 🧿 現在時刻と株価を通知
        self.notifyCurrentPrice.emit(dict_data)
        # ---------------------------------------------------------------------

    def stopProcess(self):
        """
        xlwings のインスタンスを明示的に開放する
        :return:
        """
        self.logger.info(f"{self.__class__}: stopProcess called.")

        if self.wb:
            self.wb = None  # オブジェクト参照をクリア

        # ---------------------------------------------------------------------
        # 🧿 スレッド終了シグナルの通知
        self.threadFinished.emit()
        # ---------------------------------------------------------------------


class RssConnect(QThread):
    requestWorkerInit = Signal()
    requestCurrentPrice = Signal()
    requestStopProcess = Signal()

    requestBuy = Signal(str)

    # このスレッドが開始されたことを通知するシグナル（デバッグ用など）
    threadReady = Signal()

    def __init__(self, res: AppRes, excel_path: str):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.res = res

        # excel_path = res.excel_collector
        self.worker = worker = RssConnectWorker(res, excel_path)
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

        # 現在株価を取得するメソッドへキューイング。
        self.requestCurrentPrice.connect(worker.readCurrentPrice)

        # xlwings インスタンスを破棄、スレッドを終了する下記のメソッドへキューイング。
        self.requestStopProcess.connect(worker.stopProcess)

        # Excel の関数 ExecBuy の実行
        self.requestBuy.connect(worker.executeBuy)

        # スレッド終了関連
        worker.threadFinished.connect(self.quit)  # スレッド終了時
        self.finished.connect(self.deleteLater)  # スレッドオブジェクトの削除

    def thread_ready(self):
        self.threadReady.emit()

    def run(self):
        """
        このスレッドのイベントループを開始する。
        これがなければ、スレッドはすぐに終了してしまう。
        """
        self.logger.info(
            f"{self.__class__}: run() method started. Entering event loop..."
        )
        self.exec()  # イベントループを開始
        self.logger.info(
            f"{self.__class__}: run() method finished. Event loop exited."
        )
