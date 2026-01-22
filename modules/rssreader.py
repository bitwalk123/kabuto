# Windows 固有のライブラリ
import logging
import os
import sys
import time

import pandas as pd
import xlwings as xw
from PySide6.QtCore import (
    QObject,
    Signal,
    Slot,
)

from funcs.ios import save_dataframe_to_excel
from funcs.tide import get_date_str_today
from modules.posman import PositionManager
from structs.res import AppRes

if sys.platform == "win32":
    from pywintypes import com_error


class RSSReaderWorker(QObject):
    """
    【Windows 専用】
    楽天証券マーケットスピード２ RSS が Excel シートに書き込んだ株価情報を読み取るワーカースレッド
    """
    # 銘柄名（リスト）の通知
    notifyTickerN = Signal(list, dict)
    # ティックデータを通知
    notifyCurrentPrice = Signal(dict, dict, dict)
    # 取引結果のデータフレームを通知
    notifyTransactionResult = Signal(pd.DataFrame)
    # ティックデータ保存の終了を通知
    saveCompleted = Signal(bool)
    # スレッド終了シグナル（成否の論理値）
    threadFinished = Signal(bool)

    def __init__(self, res: AppRes):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.res = res
        self.excel_path = res.excel_collector
        self._running = True

        # ---------------------------------------------------------------------
        # xlwings のインスタンス
        # この初期化プロセスでは xlwings インスタンスの初期化ができない。
        # Excel と通信する COM オブジェクトがスレッドアフィニティ（特定のCOMオブジェクトは
        # 特定のシングルスレッドアパートメントでしか動作できないという制約）を持っているため
        # ---------------------------------------------------------------------
        self.wb = None  # Excel のワークブックインスタンス
        self.sheet = None  # Excel のワークシートインスタンス

        self.max_row = None
        self.min_row = None

        # Excelシートから xlwings でデータを読み込むときの試行回数
        # 楽天証券のマーケットスピード２ RSS の書込と重なる（衝突する）と、
        # COM エラーが発生するため、リトライできるようにしている。
        self.max_retries = 3  # 最大リトライ回数
        self.retry_delay = 0.1  # リトライ間の遅延（秒）

        # Excel シートから読み取った内容をメインスレッドへ渡す作業用辞書
        self.dict_data = dict()
        self.dict_profit = dict()
        self.dict_total = dict()
        # ---------------------------------------------------------------------

        # Excel ワークシート情報
        self.cell_bottom = "------"
        self.list_code = list()  # 銘柄リスト
        self.dict_row = dict()  # 銘柄の行位置
        self.dict_name = dict()  # 銘柄名
        self.ticks = dict()  # 銘柄別データフレーム

        # Excel の列情報（VBA準拠）
        self.col_code = 1  # 銘柄コード
        self.col_name = 2  # 銘柄名
        self.col_date = 3  # 日付
        self.col_time = 4  # 時刻
        self.col_price = 5  # 現在詳細株価
        self.col_lastclose = 6  # 前日終値
        self.col_ratio = 7  # 前日比
        self.col_volume = 8  # 出来高

        # ポジション・マネージャのインスタンス
        self.posman = PositionManager()

    @Slot()
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

    @Slot()
    def initWorker(self):
        """
        スレッド開始後の初期化処理
        :return:
        """
        self.logger.info("Worker: in init process.")
        #######################################################################
        # 情報を取得する Excel ワークブック・インスタンスの生成
        self.wb = xw.Book(self.excel_path)
        self.sheet = self.wb.sheets["Cover"]
        #######################################################################
        row_max = 200  # Cover の最大行数の仮設定

        # Excel シートから、銘柄コード、銘柄名を取得
        for row in range(2, row_max + 1):
            code = self.sheet.range(row, self.col_code).value
            if code == self.cell_bottom:
                break

            self.list_code.append(code)
            self.dict_row[code] = row
            self.dict_name[code] = self.sheet.range(row, self.col_name).value

        # 株価などを一括読み取るための行範囲
        rows = list(self.dict_row.values())
        self.min_row = min(rows)
        self.max_row = max(rows)

        # 保持するティックデータの初期化 → 最後にデータフレームへ
        for code in self.list_code:
            self.ticks[code] = {"Time": [], "Price": [], "Volume": []}

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 銘柄名（リスト）などの情報を通知
        self.notifyTickerN.emit(self.list_code, self.dict_name)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # ポジションマネージャ初期化
        self.posman.initPosition(self.list_code)

    @Slot(float)
    def readCurrentPrice(self, ts: float):
        """
        現在株価の読み取り（Excel 一括読み取り版）
        :param ts: タイムスタンプ
        """
        self.dict_data.clear()
        self.dict_profit.clear()
        self.dict_total.clear()

        for attempt in range(self.max_retries):
            ###################################################################
            # 楽天証券のマーケットスピード２ RSS の書込と重なる（衝突する）と、
            # COM エラーが発生するため、リトライできるようにしている。
            try:
                # -------------------------------------------------------------
                # 株価情報を一括読み取り（列ごとに）
                # -------------------------------------------------------------
                prices = self.sheet.range((self.min_row, self.col_price), (self.max_row, self.col_price)).value
                volumes = self.sheet.range((self.min_row, self.col_volume), (self.max_row, self.col_volume)).value

                # 読み取り結果を dict_data に格納
                for i, code in enumerate(self.list_code):
                    price = prices[i]
                    volume = volumes[i]
                    if price > 0:
                        self.dict_data[code] = (ts, price, volume)
                        self.dict_profit[code] = self.posman.getProfit(code, price)
                        self.dict_total[code] = self.posman.getTotal(code)
                break
            except com_error as e:
                # -------------------------------------------------------------
                # com_error は Windows 固有
                # -------------------------------------------------------------
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
            except TypeError as e:
                self.logger.error(f"{__name__} TypeError occurred (likely 2D→1D issue): {e}")
                # リトライせず break して次の処理へ
                break
            except Exception as e:
                # -------------------------------------------------------------
                # その他のエラー
                # -------------------------------------------------------------
                self.logger.exception(f"{__name__} unexpected error during bulk read: {e}")
                raise
            #
            ###################################################################

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 現在時刻と株価を通知
        self.notifyCurrentPrice.emit(self.dict_data, self.dict_profit, self.dict_total)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # ティックデータを蓄積
        for code in self.list_code:
            if code in self.dict_data:
                ts, price, volume = self.dict_data[code]
                d = self.ticks[code]
                d["Time"].append(ts)
                d["Price"].append(price)
                d["Volume"].append(volume)

    def saveDataFrame(self):
        """
        最後にティックデータを保存する処理
        :return:
        """
        # 保存するファイル名
        date_str = get_date_str_today()
        name_excel = os.path.join(
            self.res.dir_collection,
            f"ticks_{date_str}.xlsx"
        )

        r = 0
        dict_df = dict()  # 銘柄コード別にデータフレームを保存
        for code in self.list_code:
            df = pd.DataFrame(self.ticks[code])
            r += len(df)
            # 保存する Excel では code がシート名になる → 辞書で渡す
            dict_df[code] = df

        if r == 0:
            # データフレームの総行数が 0 の場合は保存しない。
            self.logger.info(f"{__name__} データが無いため {name_excel} への保存はキャンセルされました。")
            flag = False
        else:
            # ティックデータの保存処理
            try:
                save_dataframe_to_excel(name_excel, dict_df)
                self.logger.info(f"{__name__} データが {name_excel} に保存されました。")
                flag = True
            except ValueError as e:
                self.logger.error(f"{__name__} error occurred!: {e}")
                flag = False

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 保存の終了を通知
        self.saveCompleted.emit(flag)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def stop(self):
        self._running = False

    @Slot()
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
