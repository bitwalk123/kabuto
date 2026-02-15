# Windows 固有のライブラリ
import logging
import os
import sys
import time
from typing import Optional, Callable

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
from structs.app_enum import ActionType
from structs.res import AppRes

if sys.platform == "win32":
    from pywintypes import com_error


class RSSReaderWorker(QObject):
    """
    【Windows 専用】
    楽天証券マーケットスピード２ RSS が Excel シートに書き込んだ株価情報を読み取るワーカースレッド
    """
    # 1. 銘柄名（リスト）の通知
    notifyTickerN = Signal(list, dict)
    # 2. ティックデータを通知
    notifyCurrentPrice = Signal(dict, dict, dict)
    # 3. 取引結果のデータフレームを通知
    notifyTransactionResult = Signal(pd.DataFrame)
    # 4. 約定確認結果を通知
    sendResult = Signal(str, bool)
    # 5. ティックデータ保存の終了を通知（本番用）
    saveCompleted = Signal(bool)
    # 6. データ準備完了（デバッグ用 - 本番用ではダミー）
    notifyDataReady = Signal(bool)
    # 7. スレッド終了シグナル（成否の論理値）
    threadFinished = Signal(bool)

    def __init__(self, res: AppRes) -> None:
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.prefix = self.__class__.__name__
        self.res = res
        self.excel_path = res.excel_collector
        self._running = True

        # ---------------------------------------------------------------------
        # xlwings のインスタンス
        # この初期化プロセスでは xlwings インスタンスの初期化ができない。
        # Excel と通信する COM オブジェクトがスレッドアフィニティ（特定のCOMオブジェクトは
        # 特定のシングルスレッドアパートメントでしか動作できないという制約）を持っているため
        # ---------------------------------------------------------------------
        self.wb: Optional[xw.Book] = None  # Excel のワークブックインスタンス
        self.sheet: Optional[xw.Sheet] = None  # Excel のワークシートインスタンス
        self.clear_logs: Optional[Callable] = None
        self.do_buy: Optional[Callable] = None
        self.do_sell: Optional[Callable] = None
        self.do_repay: Optional[Callable] = None
        self.is_position_present: Optional[Callable] = None

        self.max_row: Optional[int] = None
        self.min_row: Optional[int] = None

        # Excelシートから xlwings でデータを読み込むときの試行回数
        # 楽天証券のマーケットスピード２ RSS の書込と重なる（衝突する）と、
        # COM エラーが発生するため、リトライできるようにしている。
        self.max_retries = 5  # 最大リトライ回数
        self.retry_delay = 0.1  # リトライ間の遅延（秒）
        self.sec_sleep = 2  # 約定確認用のスリープ時間（秒）

        # Excel シートから読み取った内容をメインスレッドへ渡す作業用辞書
        self.dict_data: dict[str, tuple[float, float, float]] = {}
        self.dict_profit: dict[str, float] = {}
        self.dict_total: dict[str, float] = {}
        # ---------------------------------------------------------------------

        # Excel ワークシート情報
        self.cell_bottom = "------"
        self.list_code: list[str] = []  # 銘柄リスト
        self.dict_row: dict[str, int] = {}  # 銘柄の行位置
        self.dict_name: dict[str, str] = {}  # 銘柄名
        self.ticks: dict[str, dict[str, list[float]]] = {}  # 銘柄別データフレーム

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
    def getTransactionResult(self) -> None:
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
    def initWorker(self) -> None:
        """
        スレッド開始後の初期化処理
        :return:
        """
        self.logger.info("Worker: in init process.")
        #######################################################################
        # 情報を取得する Excel ワークブック・インスタンスの生成
        self.wb = wb = xw.Book(self.excel_path)
        self.sheet = wb.sheets["Cover"]
        self.clear_logs = wb.macro("ClearLogs")
        self.do_buy = wb.macro("DoBuy")
        self.do_sell = wb.macro("DoSell")
        self.do_repay = wb.macro("DoRepay")
        self.is_position_present = wb.macro("IsPositionPresent")
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
        # 古いログをクリア
        self.macro_clear_logs()

    @Slot(float)
    def readCurrentPrice(self, ts: float) -> None:
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
                prices: list[float | None] = self.sheet.range(
                    (self.min_row, self.col_price),
                    (self.max_row, self.col_price)
                ).value
                volumes: list[float | None] = self.sheet.range(
                    (self.min_row, self.col_volume),
                    (self.max_row, self.col_volume)
                ).value

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

    def saveDataFrame(self) -> None:
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
        dict_df: dict[str, pd.DataFrame] = {}  # 銘柄コード別にデータフレームを保存
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

    def stop(self) -> None:
        self._running = False

    @Slot()
    def stopProcess(self) -> None:
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

    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
    # 取引ボタンがクリックされた時など　VBAマクロとやり取りをする処理
    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
    @Slot()
    def macro_clear_logs(self) -> None:
        if sys.platform != "win32":
            self.logger.info(f"{self.prefix}: ClearLogs: 非Windows 上では実行できません。")
            return
        try:
            self.clear_logs()
            self.logger.info(f"{self.prefix}: ClearLogs completed")
        except com_error as e:
            self.logger.error(f"{self.prefix}: ClearLogs failed: {e}")
        except Exception as e:
            self.logger.exception(f"{self.prefix}: Unexpected error in ClearLogs: {e}")

    @Slot(str, float, float, str)
    def macro_do_buy(self, code: str, ts: float, price: float, note: str) -> None:
        try:
            result = self.do_buy(code)
            self.logger.info(f"{self.prefix}: DoBuy returned {result}")
        except com_error as e:
            self.logger.error(f"{self.prefix}: DoBuy failed for code={code}: {e}")
            self.sendResult.emit(code, False)
            return
        except Exception as e:
            self.logger.exception(f"{self.prefix}: Unexpected error in DoBuy: {e}")
            self.sendResult.emit(code, False)
            return

        # 注文結果が False の場合はここで終了
        if not result:
            self.sendResult.emit(code, False)
            return
        # 約定後、買建では建玉一覧に銘柄コードあり (True)
        expected_state = True

        # 約定確認
        if self.confirm_execution(code, expected_state):
            # 買建で新規建玉
            self.posman.openPosition(code, ts, price, ActionType.BUY, note)

    @Slot(str, float, float, str)
    def macro_do_sell(self, code: str, ts: float, price: float, note: str) -> None:
        try:
            result = self.do_sell(code)
            self.logger.info(f"{self.prefix}: DoSell returned {result}")
        except com_error as e:
            self.logger.error(f"{self.prefix}: DoSell failed for code={code}: {e}")
            self.sendResult.emit(code, False)
            return
        except Exception as e:
            self.logger.exception(f"{self.prefix}: Unexpected error in DoSell: {e}")
            self.sendResult.emit(code, False)
            return

        # 注文結果が False の場合はここで終了
        if not result:
            self.sendResult.emit(code, False)
            return
        # 約定後、売建では建玉一覧に銘柄コードあり (True)
        expected_state = True
        # 約定確認
        if self.confirm_execution(code, expected_state):
            # 売建で新規建玉
            self.posman.openPosition(code, ts, price, ActionType.SELL, note)

    @Slot(str, float, float, str)
    def macro_do_repay(self, code: str, ts: float, price: float, note: str) -> None:
        try:
            result = self.do_repay(code)
            self.logger.info(f"{self.prefix}: DoRepay returned {result}")
        except com_error as e:
            self.logger.error(f"{self.prefix}: DoRepay failed for code={code}: {e}")
            self.sendResult.emit(code, False)
            return
        except Exception as e:
            self.logger.exception(f"{self.prefix}: Unexpected error in DoRepay: {e}")
            self.sendResult.emit(code, False)
            return

        # 注文結果が False の場合はここで終了
        if not result:
            self.sendResult.emit(code, False)
            return
        # 約定後、返済では建玉一覧に銘柄コードなし (False)
        expected_state = False

        # 約定確認
        if self.confirm_execution(code, expected_state):
            # 建玉返済
            self.posman.closePosition(code, ts, price, note)

    def confirm_execution(self, code: str, expected_state: bool) -> bool:
        # 約定確認
        for attempt in range(self.max_retries):
            time.sleep(self.sec_sleep)
            try:
                current = bool(self.is_position_present(code))  # 論理値が返ってくるはずだけど保険に
                if current == expected_state:
                    self.logger.info(f"{self.prefix}: 約定が反映されました (attempt {attempt + 1}).")
                    self.sendResult.emit(code, True)
                    return True
                else:
                    self.logger.info(
                        f"{self.prefix}: 約定未反映 (attempt {attempt + 1}): "
                        f"is_position_present={current}, expected={expected_state}"
                    )
            except com_error as e:
                self.logger.error(f"{self.prefix}: IsPositionPresent failed for code={code}: {e}")
                self.logger.info(f"{self.prefix}: retrying... (Attempt {attempt + 1}/{self.max_retries})")
            except Exception as e:
                self.logger.exception(f"{self.prefix}: Unexpected error in IsPositionPresent: {e}")

        # self.max_retries 回確認しても変化なし → 注文未反映
        self.logger.info(f"{self.prefix}: 約定を確認できませんでした。")
        self.sendResult.emit(code, False)

        return False
