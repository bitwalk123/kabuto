import logging
import sys

import xlwings as xw

from funcs.conv import get_ticker_as_string

# Windows 固有のライブラリ
if sys.platform == "win32":
    from pywintypes import com_error

from PySide6.QtCore import QObject, QThread, Signal

from structs.res import AppRes


class PortfolioWorker(QObject):
    # 銘柄名（リスト）の通知
    notifyTickerN = Signal(list, dict)
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
        self.cell_bottom = "--------"
        self.list_ticker = list()  # 銘柄リスト
        self.dict_row = dict()  # 銘柄の行位置
        self.dict_name = dict()  # 銘柄名
        # self.dict_df = dict()  # 銘柄別データフレーム

        # Excel の列情報
        self.col_code = 0  # 銘柄コード
        self.col_name = 1  # 銘柄名称
        self.col_profit = 11  # 評価損益額
        self.col_profit_ratio = 12  # 評価損益率

    def initWorker(self):
        self.logger.info("Worker: in init process.")
        #######################################################################
        # 情報を取得する Excel ワークブック・インスタンスの生成
        self.wb = wb = xw.Book(self.excel_path)
        name_sheet = "Portfolio"
        self.sheet = wb.sheets[name_sheet]
        #
        #######################################################################

        row = 1
        flag_loop = True
        self.list_ticker = list()  # 銘柄リスト
        self.dict_row = dict()  # 銘柄の行位置
        self.dict_name = dict()  # 銘柄名
        while True:
            # 銘柄コードを強制的に文字列にする
            val = self.sheet[row, self.col_code].value
            ticker = get_ticker_as_string(val)
            # 終端判定
            if ticker == self.cell_bottom:
                # flag_loop = False
                break
            else:
                # 銘柄コード
                self.list_ticker.append(ticker)

                # 行位置
                self.dict_row[ticker] = row

                # 銘柄名
                name = self.sheet[row, self.col_name].value
                self.dict_name[ticker] = name

                # 行番号のインクリメント
                row += 1

        # --------------------------------------------------------------
        # 🧿 銘柄名などの情報を通知
        self.notifyTickerN.emit(self.list_ticker, self.dict_name)
        # --------------------------------------------------------------

    """
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
                    if price > 0:
                        # ここでもタイムスタンプを時刻に採用する
                        df.at[row, "Time"] = ts
                        df.at[row, "Price"] = price
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
    """

    def stopProcess(self):
        """
        xlwings のインスタンスを明示的に開放する
        :return:
        """
        self.logger.info("{__name__} PortfolioWorker: stopProcess called.")

        if self.wb:
            self.wb = None  # オブジェクト参照をクリア

        # -------------------------------
        # 🧿 スレッド終了シグナルの通知
        self.threadFinished.emit()
        # -------------------------------


class Portfolio(QThread):
    requestWorkerInit = Signal()
    requestCurrentPrice = Signal()
    requestStopProcess = Signal()

    # このスレッドが開始されたことを通知するシグナル（デバッグ用など）
    threadReady = Signal()

    def __init__(self, res: AppRes):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.res = res

        excel_path = res.excel_portfolio
        self.worker = worker = PortfolioWorker(res, excel_path)
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
        # self.requestCurrentPrice.connect(worker.readCurrentPrice)

        # xlwings インスタンスを破棄、スレッドを終了する下記のメソッドへキューイング。
        self.requestStopProcess.connect(worker.stopProcess)

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
            f"{__name__} Portfolio: run() method started. Entering event loop..."
        )
        self.exec()  # イベントループを開始
        self.logger.info(
            f"{__name__} Portfolio: run() method finished. Event loop exited."
        )
