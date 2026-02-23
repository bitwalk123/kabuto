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

from PySide6.QtCore import QObject, QThread, Signal, Slot

from structs.res import AppRes


class StockCollectorWorker(QObject):
    # 銘柄名（リスト）の通知
    notifyTickerN = Signal(list, dict)
    # 保存の終了を通知
    saveCompleted = Signal(bool)
    # スレッドの終了を通知
    threadFinished = Signal()

    def __init__(self, res: AppRes, excel_path: str):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.res = res
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
        self.col_ratio = 6  # 前日比
        self.col_volume = 7  # 出来高

    def initWorkerOld(self):
        self.logger.info("Worker: in init process.")
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
                    "Price": list(),
                    "Volume": list(),
                })

                # 行番号のインクリメント
                row += 1

        # --------------------------------------------------------------
        # 🧿 銘柄名などの情報を通知
        self.notifyTickerN.emit(self.list_ticker, self.dict_name)
        # --------------------------------------------------------------

    @Slot()
    def initWorker(self):
        self.logger.info("Worker: in init process.")

        #######################################################################
        # Excel ワークブックとシートの取得
        self.wb = wb = xw.Book(self.excel_path)
        self.sheet = sheet = wb.sheets["Cover"]
        #######################################################################

        # 読み取り上限（必要に応じて調整可能）
        max_row = 200

        # 銘柄コード列と銘柄名列を一括取得（COM 呼び出し最小化）
        values_code = sheet.range((1, self.col_code), (max_row, self.col_code)).value
        values_name = sheet.range((1, self.col_name), (max_row, self.col_name)).value

        # 初期化
        self.list_ticker.clear()
        self.dict_row.clear()
        self.dict_name.clear()
        self.dict_df.clear()

        # 行ごとに処理
        for row, (ticker, name) in enumerate(zip(values_code, values_name), start=1):
            # デリミタ検出で終了
            if ticker == self.cell_bottom:
                break
            # 空行はスキップ（壊れにくさ向上）
            if not ticker:
                continue
            # 銘柄コード
            self.list_ticker.append(ticker)
            # 行位置
            self.dict_row[ticker] = row
            # 銘柄名
            self.dict_name[ticker] = name
            # 空のティックデータ DataFrame を準備
            self.dict_df[ticker] = pd.DataFrame({
                "Time": [],
                "Price": [],
                "Volume": [],
            })
        self.logger.info(f"Worker: {len(self.list_ticker)} tickers loaded.")
        # --------------------------------------------------------------
        # 🧿 銘柄名などの情報を通知
        self.notifyTickerN.emit(self.list_ticker, self.dict_name)
        # --------------------------------------------------------------

    def readCurrentPrice(self):
        for ticker in self.list_ticker:
            row_excel = self.dict_row[ticker]
            df = self.dict_df[ticker]
            row = len(df)
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
                    volume = self.sheet[row_excel, self.col_volume].value
                    if price > 0:
                        # ここでもタイムスタンプを時刻に採用する
                        df.at[row, "Time"] = ts
                        df.at[row, "Price"] = price
                        df.at[row, "Volume"] = volume
                        # print(ticker, ts, price)
                    break
                except com_error as e:
                    # ---------------------------------------------------------
                    # com_error は Windows 固有
                    # ---------------------------------------------------------
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
                #
                ###############################################################

    def saveDataFrame(self):
        # 保存するファイル名
        date_str = get_date_str_today()
        name_excel = os.path.join(
            self.res.dir_collection,
            f"ticks_{date_str}.xlsx"
        )
        # 念のため、空のデータでないか確認して空でなければ保存
        r = 0
        for ticker in self.list_ticker:
            df = self.dict_df[ticker]
            r += len(df)
        if r == 0:
            # すべてのデータフレームの行数が 0 の場合は保存しない。
            self.logger.info(f"{__name__} データが無いため {name_excel} への保存はキャンセルされました。")
            flag = False
        else:
            # ティックデータの保存処理
            try:
                save_dataframe_to_excel(name_excel, self.dict_df)
                self.logger.info(f"{__name__} データを {name_excel} に保存しました。")
                flag = True
            except ValueError as e:
                self.logger.error(f"{__name__} error occurred!: {e}")
                flag = False

        # ----------------------------
        # 🧿 保存の終了を通知
        self.saveCompleted.emit(flag)
        # ----------------------------

    def stopProcess(self):
        """
        xlwings のインスタンスを明示的に開放する
        :return:
        """
        self.logger.info("Worker: stopProcess called.")

        if self.wb:
            """
            try:
                self.wb.close()  # ブックを閉じる
                self.logger.info("Worker: Excel book closed.")
            except Exception as e:
                self.logger.error(f"Worker: Error closing book: {e}")
            # ブックを閉じた後、その親アプリケーションも終了させる
            if self.wb.app:
                try:
                    self.wb.app.quit()
                    self.logger.info("Worker: Excel app quit.")
                except Exception as e:
                    self.logger.error(f"Worker: Error quitting app: {e}")
            """
            self.wb = None  # オブジェクト参照をクリア

        # -------------------------------
        # 🧿 スレッド終了シグナルの通知
        self.threadFinished.emit()
        # -------------------------------


class StockCollector(QThread):
    requestWorkerInit = Signal()
    requestCurrentPrice = Signal()
    requestSaveDataFrame = Signal()
    requestStopProcess = Signal()

    # このスレッドが開始されたことを通知するシグナル（デバッグ用など）
    threadReady = Signal()

    def __init__(self, res: AppRes):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.res = res

        excel_path = res.excel_collector
        self.worker = worker = StockCollectorWorker(res, excel_path)
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

        # データフレームを保存するメソッドへキューイング
        self.requestSaveDataFrame.connect(worker.saveDataFrame)

        # xlwings インスタンスを破棄、スレッドを終了する下記のメソッドへキューイング。
        self.requestStopProcess.connect(worker.stopProcess)

        # スレッド終了関連
        # worker.threadFinished.connect(self.on_thread_finished)
        worker.threadFinished.connect(self.quit)
        self.finished.connect(worker.deleteLater)
        self.finished.connect(self.deleteLater)

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
