# Windows 固有のライブラリ
import logging
import os
import sys
import time

import pandas as pd
import xlwings as xw
from PySide6.QtCore import QObject, Signal

from funcs.ios import save_dataframe_to_excel
from funcs.tide import get_date_str_today
from modules.posman import PositionManager
from structs.res import AppRes

if sys.platform == "win32":
    from pywintypes import com_error


class RSSReaderWorker(QObject):
    """
    【Windows 専用】
    楽天証券のマーケットスピード２ RSS が Excel シートに書き込んだ株価情報を読み取るワーカースレッド
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
        # self.dict_df = dict()  # 銘柄別データフレーム
        self.ticks = dict()  # 銘柄別データフレーム

        # Excel の列情報
        self.col_code = 0  # 銘柄コード
        self.col_name = 1  # 銘柄名
        self.col_date = 2  # 日付
        self.col_time = 3  # 時刻
        self.col_price = 4  # 現在詳細株価
        self.col_lastclose = 5  # 前日終値
        self.col_ratio = 6  # 前日比
        self.col_volume = 7  # 出来高

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

        # dict_name = dict()  # 銘柄名
        # dict_lastclose = dict()  # 銘柄別前日終値

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
                self.dict_name[code] = self.sheet[row, self.col_name].value

                # 前日の終値の横線
                # dict_lastclose[code] = self.sheet[row, self.col_lastclose].value

                '''
                # 銘柄別に空のデータフレームを準備
                self.dict_df[code] = pd.DataFrame({
                    "Time": list(),
                    "Price": list(),
                    "Volume": list(),
                })
                '''

                # 行番号のインクリメント
                row += 1

        # 一括読み取り対象の行範囲を取得
        rows = [self.dict_row[code] for code in self.list_code]
        self.min_row = min(rows)
        self.max_row = max(rows)

        # 銘柄別に空の辞書/リストを準備 → あとでデータフレームに変換
        for code in self.list_code:
            self.ticks[code] = {
                "Time": [],
                "Price": [],
                "Volume": [],
            }

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 銘柄名（リスト）などの情報を通知
        self.notifyTickerN.emit(self.list_code, self.dict_name)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # ポジション・マネージャの初期化
        self.posman.initPosition(self.list_code)

    def readCurrentPriceOld(self, ts: float):
        """
        現在株価の読み取り
        :param ts:
        :return:
        """
        self.dict_data.clear()
        self.dict_profit.clear()
        self.dict_total.clear()
        for code in self.list_code:
            row_excel = self.dict_row[code]
            # Excel シートから株価情報を取得
            for attempt in range(self.max_retries):
                ###############################################################
                # 楽天証券のマーケットスピード２ RSS の書込と重なる（衝突する）と、
                # COM エラーが発生するため、リトライできるようにしている。
                # -------------------------------------------------------------
                try:
                    # Excelシートから株価データを取得
                    price = self.sheet[row_excel, self.col_price].value
                    volume = self.sheet[row_excel, self.col_volume].value
                    if price > 0:
                        # ここでもタイムスタンプを時刻に採用する
                        self.dict_data[code] = (ts, price, volume)  # tuple の方が高速で軽い！
                        self.dict_profit[code] = self.posman.getProfit(code, price)
                        self.dict_total[code] = self.posman.getTotal(code)
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
        self.notifyCurrentPrice.emit(
            self.dict_data, self.dict_profit, self.dict_total
        )
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # ティックデータをまとめて保持
        for code in self.list_code:
            '''
            df = self.dict_df[code]
            row = len(df)
            '''
            # 寄っていない場合はデータが無い銘柄コードがある！
            if code in self.dict_data:
                ts, price, volume = self.dict_data[code]
                # df.loc[row] = [ts, price, volume]
                d = self.ticks[code]
                d["Time"].append(ts)
                d["Price"].append(price)
                d["Volume"].append(volume)

    def readCurrentPrice(self, ts: float):
        """
        現在株価の読み取り（Excel 一括読み取り版）
        :param ts: タイムスタンプ
        """
        self.dict_data.clear()
        self.dict_profit.clear()
        self.dict_total.clear()

        try:
            # 一括読み取り（列ごとに）
            prices = self.sheet.range((self.min_row, self.col_price), (self.max_row, self.col_price)).value
            volumes = self.sheet.range((self.min_row, self.col_volume), (self.max_row, self.col_volume)).value

            print(prices)
            # 読み取り結果を dict_data に格納
            for i, code in enumerate(self.list_code):
                price = prices[i]
                volume = volumes[i]
                if price > 0:
                    self.dict_data[code] = (ts, price, volume)
                    self.dict_profit[code] = self.posman.getProfit(code, price)
                    self.dict_total[code] = self.posman.getTotal(code)

        except com_error as e:
            self.logger.error(f"{__name__} COM error during bulk read: {e}")
            raise
        except Exception as e:
            self.logger.exception(f"{__name__} unexpected error during bulk read: {e}")
            raise

        # 🧿 GUI に通知
        self.notifyCurrentPrice.emit(self.dict_data, self.dict_profit, self.dict_total)

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
        dict_df = dict()  # 銘柄コード別にデータフレームを保存
        # 念のため、全てが空のデータでないか確認して空でなければ保存（無用な上書きを回避）
        r = 0
        for code in self.list_code:
            # df = self.dict_df[code]
            df = pd.DataFrame(self.ticks[code])
            r += len(df)
            # 保存する Excel では code がシート名になる → 辞書で渡す
            dict_df[code] = df
        if r == 0:
            # すべてのデータフレームの行数が 0 の場合は保存しない。
            self.logger.info(f"{__name__} データが無いため {name_excel} への保存はキャンセルされました。")
            flag = False
        else:
            # ティックデータの保存処理
            try:
                # save_dataframe_to_excel(name_excel, self.dict_df)
                save_dataframe_to_excel(name_excel, dict_df)
                self.logger.info(f"{__name__} データが {name_excel} に保存されました。")
                flag = True
            except ValueError as e:
                self.logger.error(f"{__name__} error occurred!: {e}")
                flag = False

        # ----------------------------
        # 🧿 保存の終了を通知
        self.saveCompleted.emit(flag)
        # ----------------------------

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
