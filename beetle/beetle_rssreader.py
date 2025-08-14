# Windows 固有のライブラリ
import logging
import sys
import time

import pandas as pd
import xlwings as xw
from PySide6.QtCore import QObject, Signal

from modules.posman import PositionManager

if sys.platform == "win32":
    from pywintypes import com_error


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
        self._running = True
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
        self.list_code = list()  # 銘柄リスト
        self.dict_row = dict()  # 銘柄の行位置

        # Excel の列情報
        self.col_code = 0  # 銘柄コード
        self.col_name = 1  # 銘柄名
        self.col_date = 2  # 日付
        self.col_time = 3  # 時刻
        self.col_price = 4  # 現在詳細株価
        self.col_lastclose = 5  # 前日終値

        # ポジション・マネージャのインスタンス
        self.posman = PositionManager()

    def getTransactionResult(self):
        """
        取引結果を取得
        :return:
        """
        df = self.posman.getTransactionResult()
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 取引結果のデータフレームを通知
        self.notifyTransactionResult.emit(df)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def initWorker(self):
        self.logger.info("Worker: in init process.")
        #######################################################################
        # 情報を取得する Excel ワークブック・インスタンスの生成
        self.wb = wb = xw.Book(self.excel_path)
        name_sheet = "Cover"
        self.sheet = wb.sheets[name_sheet]
        #
        #######################################################################

        dict_name = dict()  # 銘柄名
        dict_lastclose = dict()  # 銘柄別前日終値

        row = 1
        flag_loop = True
        while flag_loop:
            code = self.sheet[row, self.col_code].value
            if code == self.cell_bottom:
                flag_loop = False
            else:
                # 銘柄コード
                self.list_code.append(code)

                # 行位置
                self.dict_row[code] = row

                # 銘柄名
                dict_name[code] = self.sheet[row, self.col_name].value

                # 前日の終値の横線
                dict_lastclose[code] = self.sheet[row, self.col_lastclose].value

                # 行番号のインクリメント
                row += 1

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 銘柄名（リスト）などの情報を通知
        self.notifyTickerN.emit(
            self.list_code, dict_name, dict_lastclose
        )
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # ポジション・マネージャの初期化
        self.posman.initPosition(self.list_code)

    def readCurrentPrice(self):
        """
        現在株価の読み取り
        :return:
        """
        dict_data = dict()
        dict_profit = dict()
        dict_total = dict()
        for i, code in enumerate(self.list_code):
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
                        dict_data[code] = [ts, price]
                        dict_profit[code] = self.posman.getProfit(code, price)
                        dict_total[code] = self.posman.getTotal(code)
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

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 現在時刻と株価を通知
        self.notifyCurrentPrice.emit(dict_data, dict_profit, dict_total)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def stop(self):
        self._running = False

    def stopProcess(self):
        """
        xlwings のインスタンスを明示的に開放する
        :return:
        """
        self.logger.info("Worker: stopProcess called.")

        if self.wb:
            self.wb = None  # オブジェクト参照をクリア

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 スレッド終了シグナルの通知
        self.threadFinished.emit(True)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
