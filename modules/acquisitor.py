import logging
import sys
import time

import pandas as pd
import xlwings as xw

# Windows 固有のライブラリ
if sys.platform == "win32":
    from pywintypes import com_error

from modules.position_mannager import PositionManager

from PySide6.QtCore import QObject, Signal


class AcquireWorker(QObject):
    """
    【Windows 専用】
    楽天証券のマーケットスピード２ RSS が Excel シートに書き込んだ株価情報を読み取る処理をするワーカースレッド
    """
    # 銘柄名（リスト）の通知
    notifyTickerN = Signal(list, dict, dict)

    # ティックデータを通知
    notifyCurrentPrice = Signal(dict, dict, dict)

    # 取引結果のデータフレームを通知
    notifyTransactionResult = Signal(pd.DataFrame)

    # スレッド終了シグナル（成否の論理値）
    threadFinished = Signal(bool)

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

        # 銘柄リスト
        self.list_ticker = list()

        # Excel の列情報
        self.col_code = 0  # 銘柄コード
        self.col_name = 1  # 銘柄名
        self.col_date = 2  # 日付
        self.col_time = 3  # 時刻
        self.col_price = 4  # 現在詳細株価
        self.col_lastclose = 5  # 前日終値

        # 最大銘柄数
        # プログラム的に登録されている銘柄数を調べるべきだが、現在のところ 3 銘柄に固定
        self.num_max = 3

        # ポジション・マネージャのインスタンス
        self.posman = PositionManager()

        # Parabolic SAR の辞書
        # self.dict_psar = dict()

    def getTransactionResult(self):
        """
        取引結果を取得
        :return:
        """
        df = self.posman.getTransactionResult()
        self.notifyTransactionResult.emit(df)

    def loadExcel(self):
        #######################################################################
        # 情報を取得する Excel ワークブック・インスタンスの生成
        self.wb = wb = xw.Book(self.excel_path)
        name_sheet = "Cover"
        self.sheet = wb.sheets[name_sheet]

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
        # ----------------------------------------------------
        # 🧿 銘柄名（リスト）などの情報を通知
        self.notifyTickerN.emit(
            self.list_ticker, dict_name, dict_lastclose
        )
        # -----------------------------------------------

        # ポジション・マネージャの初期化
        self.posman.initPosition(self.list_ticker)

    def readCurrentPrice(self):
        """
        現在株価の読み取り
        :return:
        """
        dict_data = dict()
        dict_profit = dict()
        dict_total = dict()
        for i, ticker in enumerate(self.list_ticker):
            row = i + 1
            # Excel シートから株価情報を取得
            for attempt in range(self.max_retries):
                ###############################################################
                # 楽天証券のマーケットスピード２ RSS の書込と重なる（衝突する）と、
                # COM エラーが発生するため、リトライできるようにしている。
                # -------------------------------------------------------------
                try:
                    ts = time.time()
                    # Excelシートから株価データを取得
                    price = self.sheet[row, self.col_price].value
                    if price > 0:
                        # ここでもタイムスタンプを時刻に採用する
                        dict_data[ticker] = [ts, price]
                        dict_profit[ticker] = self.posman.getProfit(ticker, price)
                        dict_total[ticker] = self.posman.getTotal(ticker)
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

        # -------------------------------------------
        # 🧿 現在時刻と株価を通知
        self.notifyCurrentPrice.emit(
            dict_data, dict_profit, dict_total
        )
        # -------------------------------------------

    def stopProcess(self):
        """
        xlwings のインスタンスを明示的に開放する
        :return:
        """
        if self.wb:
            """
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
            """
            self.wb = None  # オブジェクト参照をクリア
        # -------------------------
        # 🧿 スレッド終了シグナルの通知
        self.threadFinished.emit(True)
        # -------------------------
