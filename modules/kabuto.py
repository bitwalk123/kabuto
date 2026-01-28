import logging
import os
import time

import pandas as pd
from PySide6.QtCore import (
    QThread,
    QTimer,
    Qt,
    Signal,
)
from PySide6.QtGui import (
    QCloseEvent,
    QIcon,
)
from PySide6.QtWidgets import (
    QDialog,
    QMainWindow,
    QSizePolicy,
)

from funcs.conv import conv_transaction_df2html
from funcs.setting import update_setting
from funcs.tide import get_intraday_timestamp
from funcs.tse import get_ticker_name_list
from funcs.uis import clear_boxlayout
from modules.dock import DockTrader
from structs.app_enum import ActionType
from modules.reviewer import ExcelReviewWorker
from modules.rssreader import RSSReaderWorker
from widgets.dialogs import DlgAboutThis, DlgCodeSel
from widgets.statusbars import StatusBar
from widgets.toolbars import ToolBar
from modules.trader import Trader
from modules.win_transaction import WinTransaction
from structs.res import AppRes
from widgets.containers import ScrollArea, Widget
from widgets.layouts import VBoxLayout


class Kabuto(QMainWindow):
    __app_name__ = "Kabuto"
    __version__ = "0.2.10"
    __author__ = "Fuhito Suguri"
    __license__ = "MIT"

    # ワーカーの初期化シグナル
    requestWorkerInit = Signal()

    # 現在価格取得リクエスト・シグナル
    requestCurrentPrice = Signal(float)
    requestSaveDataFrame = Signal()
    requestStopProcess = Signal()

    # 売買
    requestPositionOpen = Signal(str, float, float, ActionType, str)
    requestPositionClose = Signal(str, float, float, str)
    requestTransactionResult = Signal()

    # このスレッドが開始されたことを通知するシグナル（デバッグ用など）
    threadReady = Signal()

    def __init__(self, debug: bool = True):
        super().__init__()
        self.logger = logging.getLogger(__name__)  # モジュール固有のロガーを取得
        self.res = res = AppRes()
        res.debug = debug  # デバッグ・モードを保持
        #######################################################################
        # リアルタイム / デバッグ モード固有の設定
        if debug:
            # デバッグ・モード
            self.logger.info(f"{__name__}: デバッグモードで起動しました。")
            self.timer_interval = 100  # タイマー間隔（ミリ秒）（デバッグ時）
            self.flag_data_ready = False
        else:
            # リアルタイム・モード
            self.logger.info(f"{__name__}: 通常モードで起動しました。")
            # self.timer_interval = 1000  # タイマー間隔（ミリ秒）
            self.timer_interval = 2000  # タイマー間隔（ミリ秒）
        #
        #######################################################################
        # ---------------------------------------------------------------------
        # 株価取得スレッド用インスタンス
        # ---------------------------------------------------------------------
        self.thread = QThread(self)
        self.worker = None
        # ---------------------------------------------------------------------
        # Trader インスタンス
        # 銘柄コード別にチャートや売買情報および売買機能の UI を提供する
        # ---------------------------------------------------------------------
        self.trader: Trader | None = None
        # インスタンスを保持する辞書
        self.dict_trader = dict()
        # 銘柄コードの全リスト
        self.list_code = list()
        # 選択した銘柄コードのリスト
        self.list_code_selected = list()
        # ---------------------------------------------------------------------
        # 取引履歴
        # ---------------------------------------------------------------------
        # 取引明細用データフレーム
        self.df_transaction = None
        # 取引明細用ダイアログ・インスタンス
        self.win_transaction: WinTransaction | None = None
        # ---------------------------------------------------------------------
        # 時刻関連
        # ---------------------------------------------------------------------
        # システム時刻（タイムスタンプ形式）
        self.ts_system = 0
        # ザラ場の開始時間などのタイムスタンプ取得（本日分）
        self.dict_ts = get_intraday_timestamp()
        # ---------------------------------------------------------------------
        # 取引が終了したかどうかのフラグ
        self.finished_trading = False
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        #  UI
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # ウィンドウアイコンとタイトルを設定
        self.setWindowIcon(QIcon(os.path.join(res.dir_image, "kabuto.png")))
        title_win = f"{self.__app_name__} - {self.__version__}"
        if debug:
            # デバッグモードを示す文字列を追加
            title_win = f"{title_win} [debug mode]"
        self.setWindowTitle(title_win)
        # ---------------------------------------------------------------------
        # ツールバー
        # ---------------------------------------------------------------------
        self.toolbar = toolbar = ToolBar(res)
        toolbar.clickedAbout.connect(self.on_about)
        toolbar.clickedPlay.connect(self.on_review_play)
        toolbar.clickedStop.connect(self.on_review_stop)
        toolbar.clickedTransaction.connect(self.on_show_transaction)
        toolbar.selectedExcelFile.connect(self.on_create_thread_review)
        self.addToolBar(toolbar)
        # ---------------------------------------------------------------------
        # ステータスバー
        # ---------------------------------------------------------------------
        self.statusbar = statusbar = StatusBar(res)
        self.setStatusBar(statusbar)
        # ---------------------------------------------------------------------
        # メイン・ウィジェット
        # ---------------------------------------------------------------------
        self.area_chart = sa = ScrollArea()
        self.setCentralWidget(sa)
        # ベース・ウィジェット
        base = Widget()
        base.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed
        )
        sa.setWidget(base)
        self.layout = layout = VBoxLayout()
        layout.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        base.setLayout(layout)
        # ---------------------------------------------------------------------
        # タイマー
        # ---------------------------------------------------------------------
        self.timer = timer = QTimer()
        timer.setInterval(self.timer_interval)
        if debug:
            # デバッグモードではファイルを読み込んでからスレッドを起動
            timer.timeout.connect(self.on_request_data_review)
        else:
            # リアルタイムモードでは、直ちにスレッドを起動
            timer.timeout.connect(self.on_request_data)
            # RSS用Excelファイルを指定してxlwingsを利用するスレッド
            self.on_create_thread()

    def closeEvent(self, event: QCloseEvent):
        """
        アプリ終了イベント
        :param event:
        :return:
        """
        # ---------------------------------------------------------------------
        # タイマーの停止
        # ---------------------------------------------------------------------
        if self.timer.isActive():
            self.timer.stop()
            self.logger.info(f"{__name__}: タイマーを停止しました。")
        # ---------------------------------------------------------------------
        # self.thread スレッドの削除
        # ---------------------------------------------------------------------
        try:
            if self.thread.isRunning():
                self.requestStopProcess.emit()
                time.sleep(1)

            if self.thread is not None:
                self.thread.quit()
                self.thread.wait()
                self.logger.info(f"{__name__}: スレッド self.thread を削除しました。")

            if self.worker is not None:
                self.worker.deleteLater()
                self.worker = None
                self.logger.info(f"{__name__}: ワーカー self.worker を削除しました。")

            if self.thread is not None:
                self.thread.deleteLater()
                self.thread = None
        except RuntimeError as e:
            self.logger.error(f"{__name__}: 終了時にエラー発生: {e}")
        # ---------------------------------------------------------------------
        self.logger.info(f"{__name__} 停止して閉じました。")
        event.accept()

    def create_trader(self, dict_name: dict):
        """
        選択した銘柄数分の Trader インスタンスの生成
        （リアルタイム・モード、デバッグ・モード共通）
        :param dict_name:
        :return:
        """
        # 配置済みの Trader インスタンスを消去
        clear_boxlayout(self.layout)
        # Trader 辞書のクリア
        self.dict_trader = dict()
        # ---------------------------------------------------------------------
        # 選択した銘柄数分の Trader および Ticker インスタンスの生成
        # ---------------------------------------------------------------------
        for code in self.list_code_selected:
            update_setting(self.res, code)
            # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
            # Trader インスタンスの生成
            # 主にチャート表示用（選択された銘柄コードのみ）
            # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
            self.trader = trader = Trader(self.res, code, self.dict_ts)
            # Dock の売買ボタンのクリック・シグナルを直接ハンドリング
            trader.dock.clickedBuy.connect(self.on_buy)
            trader.dock.clickedRepay.connect(self.on_repay)
            trader.dock.clickedSell.connect(self.on_sell)
            # Trader 辞書に保持
            self.dict_trader[code] = trader
            # 「銘柄名　(code)」をタイトルにして設定し直し
            trader.setChartTitle(f"{dict_name[code]} ({code})")
            # 当日ザラ場時間（x軸の範囲設定）
            # trader.setTimeAxisRange(self.dict_ts["start"], self.dict_ts["end"])
            # 前日終値
            # if dict_lastclose[code] > 0:
            #    trader.setLastCloseLine(dict_lastclose[code])
            # 配置
            self.layout.addWidget(trader)
        # ---------------------------------------------------------------------
        # チャートエリアの面積を更新
        # ---------------------------------------------------------------------
        self.area_chart.setMinimumWidth(self.res.trend_width)
        n = len(self.list_code_selected)
        if self.res.trend_n_max < n:
            n = self.res.trend_n_max
        self.area_chart.setFixedHeight(self.res.trend_height * n + 4)

    def force_closing_position(self):
        self.logger.info(f"{__name__} 売買を強制終了します。")
        for code in self.dict_trader.keys():
            trader: Trader = self.dict_trader[code]
            dock: DockTrader = trader.dock
            dock.forceStopAutoPilot()

    def get_current_tick_data(self) -> dict:
        """
        チャートが保持しているティックデータをデータフレームで取得
        :return:
        """
        traders = self.dict_trader
        return {code: t.getTimePrice() for code, t in traders.items()}

    def on_about(self):
        """
        このアプリについて（ダイアログ表示）
        :return:
        """
        DlgAboutThis(
            self.res,
            self.__app_name__,
            self.__version__,
            self.__author__,
            self.__license__,
            "kabuto.png",
        ).exec()

    def on_create_thread(self):
        """
        リアルタイム用ティックデータ取得スレッドの生成
        :return:
        """
        # ---------------------------------------------------------------------
        # 00. リアルタイム用データ取得インスタンスの生成
        self.worker = worker = RSSReaderWorker(self.res)
        worker.moveToThread(self.thread)
        # ---------------------------------------------------------------------
        # 01. データ読み込み済みの通知（レビュー用のみ）
        # リアルタイム用には本機能なし
        # =====================================================================
        # 02. スレッドが開始されたら、ワーカースレッド内で初期化処理を実行するシグナルを発行
        self.thread.started.connect(self.requestWorkerInit.emit)
        # ---------------------------------------------------------------------
        # 03. 初期化処理は主に xlwings 関連処理
        self.requestWorkerInit.connect(worker.initWorker)
        # ---------------------------------------------------------------------
        # 04. 売買ポジション処理用のメソッドへキューイング
        self.requestPositionOpen.connect(worker.posman.openPosition)
        self.requestPositionClose.connect(worker.posman.closePosition)
        # ---------------------------------------------------------------------
        # 05. 取引結果を取得するメソッドへキューイング
        self.requestTransactionResult.connect(worker.getTransactionResult)
        # ---------------------------------------------------------------------
        # 06. 現在株価を取得するメソッドへキューイング。
        self.requestCurrentPrice.connect(worker.readCurrentPrice)
        # ---------------------------------------------------------------------
        # 07. データフレームを保存するメソッドへキューイング
        self.requestSaveDataFrame.connect(worker.saveDataFrame)
        # ---------------------------------------------------------------------
        # 08. スレッドを終了する下記のメソッドへキューイング（リアルタイムでは xlwings 関連）。
        self.requestStopProcess.connect(worker.stopProcess)
        # =====================================================================
        # 10. 初期化後の銘柄情報を通知
        worker.notifyTickerN.connect(self.on_create_trader)
        # ---------------------------------------------------------------------
        # 11. タイマーで現在時刻と株価を通知
        worker.notifyCurrentPrice.connect(self.on_update_data)
        # ---------------------------------------------------------------------
        # 12. 取引結果を通知
        worker.notifyTransactionResult.connect(self.on_transaction_result)
        # ---------------------------------------------------------------------
        # 13. データフレームの保存終了を通知
        worker.saveCompleted.connect(self.on_save_completed)
        # ---------------------------------------------------------------------
        # 19. スレッド終了関連
        worker.threadFinished.connect(self.on_thread_finished)
        # =====================================================================
        # 20. スレッドを開始
        self.thread.start()

    def on_create_trader(self, list_code: list, dict_name: dict):
        """
        Trader インスタンスの生成（リアルタイム）
        :param list_code:
        :param dict_name:
        :return:
        """
        self.list_code = list_code
        if self.res.debug:
            # -----------------------------------------------------------------
            # 選択された銘柄数分の Trader インスタンスの生成
            # -----------------------------------------------------------------
            self.create_trader(dict_name)
            # -----------------------------------------------------------------
            # デバッグの場合はスタート・ボタンがクリックされるまでは待機
            # -----------------------------------------------------------------
            self.logger.info(f"{__name__}: レビューの準備ができました。")
            return

        # ---------------------------------------------------------------------
        # Excel から読み取った銘柄を標準出力（確認用）
        # ---------------------------------------------------------------------
        self.logger.info(f"{__name__}: ティックデータ収集銘柄一覧")
        for code in list_code:
            self.logger.info(f"{__name__}: {code}, {dict_name[code]}")
        # ---------------------------------------------------------------------
        # 銘柄コードに対応する銘柄名の取得
        # ---------------------------------------------------------------------
        dict_name = get_ticker_name_list(list_code)
        # 「銘柄名 (銘柄コード)」の文字列リスト
        list_ticker = [f"{dict_name[code]} ({code})" for code in dict_name.keys()]
        # ---------------------------------------------------------------------
        # シミュレーション対象の銘柄を選択するダイアログ
        # ---------------------------------------------------------------------
        # デフォルトの銘柄コードの要素のインデックス
        idx_default = list_code.index(self.res.code_default)
        dlg_code = DlgCodeSel(self.res, list_ticker, idx_default)
        if dlg_code.exec() == QDialog.DialogCode.Accepted:
            # -----------------------------------------------------------------
            # 選択された銘柄のみデータ収集＋自動売買する。他はデータ収集のみ
            # -----------------------------------------------------------------
            self.list_code_selected = [list_code[r] for r in dlg_code.getSelected()]
            # -----------------------------------------------------------------
            # 選択された銘柄数分の Trader インスタンスの生成
            # -----------------------------------------------------------------
            self.create_trader(dict_name)
            # -----------------------------------------------------------------
            # リアルタイムの場合はここでタイマーを開始
            # -----------------------------------------------------------------
            self.timer.start()
            self.logger.info(f"{__name__}: タイマーを開始しました。")

    def on_request_data(self):
        """
        タイマー処理（リアルタイム）
        """
        # システム時刻
        self.ts_system = time.time()
        if self.dict_ts["start"] <= self.ts_system <= self.dict_ts["end_1h"]:
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # 🧿 現在価格の取得要求をワーカースレッドに通知
            self.requestCurrentPrice.emit(self.ts_system)
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        elif self.dict_ts["start_2h"] <= self.ts_system <= self.dict_ts["end_2h"]:
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # 🧿 現在価格の取得要求をワーカースレッドに通知
            self.requestCurrentPrice.emit(self.ts_system)
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        elif self.dict_ts["end_2h"] < self.ts_system <= self.dict_ts["ca"]:
            if not self.finished_trading:
                # ポジションがあればクローズする
                self.force_closing_position()
                self.finished_trading = True
        elif self.dict_ts["ca"] < self.ts_system:
            self.timer.stop()
            self.logger.info(f"{__name__}: タイマーを停止しました。")
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # 🧿 取引結果を取得
            self.requestTransactionResult.emit()
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # 🧿 収集したデータの保存
            self.requestSaveDataFrame.emit()
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        else:
            pass

        # ツールバーの時刻を更新
        self.toolbar.updateTime(self.ts_system)

    def on_save_completed(self, state: bool):
        if state:
            self.logger.info("ティック・データを正常に保存しました。")
        else:
            self.logger.info("ティック・データを正常に保存できませんでした。")

    def on_show_transaction(self):
        """
        取引明細の表示
        :return:
        """
        self.win_transaction = WinTransaction(self.res, self.df_transaction)
        self.win_transaction.show()

    def on_thread_finished(self, result: bool):
        """
        スレッド終了時のログ
        :param result:
        :return:
        """
        if result:
            self.logger.info(f"{__name__}: スレッドが正常終了しました。")
        else:
            self.logger.error(f"{__name__}: スレッドが異常終了しました。")
        # タイマーの停止
        if self.timer.isActive():
            self.timer.stop()
            self.logger.info(f"{__name__}: タイマーを停止しました。")

    def on_ticker_ready(self, code: str):
        self.logger.info(f"{__name__}: 銘柄コード {code} のスレッドの準備ができました。")

    def on_transaction_result(self, df: pd.DataFrame):
        """
        取引結果のデータフレームを取得（リアルタイム、デバッグ・モード共通）
        :param df:
        :return:
        """
        # 取引明細を標準出力
        print(df)
        print("合計損益", df["損益"].sum())
        # ---------------------------------------------------------------------
        # 取引明細の保存
        # ---------------------------------------------------------------------
        html_trans = f"{self.dict_ts["datetime_str"]}.html"
        path_trans = os.path.join(self.res.dir_transaction, html_trans)
        # 取引明細を HTML（リスト）へ変換
        list_html = conv_transaction_df2html(df)
        with open(path_trans, mode="w", encoding="utf_8") as f:
            f.write('\n'.join(list_html))  # リストを改行文字で連結
        self.logger.info(f"{__name__}: 取引明細が {path_trans} に保存されました。")
        # インスタンス変数に取引明細を保持
        self.df_transaction = df
        # ツールバーの「取引履歴」ボタンを Enabled にする
        self.toolbar.set_transaction()

    def on_update_data(self, dict_data: dict, dict_profit: dict, dict_total: dict):
        """
        ティックデータ、含み益、損益の更新
        :param dict_data:
        :param dict_profit:
        :param dict_total:
        :return:
        """
        # 受け取った瞬間にコピー
        # 受け取った辞書はスレッド側で使い回しているため
        dict_data = dict_data.copy()
        dict_profit = dict_profit.copy()
        dict_total = dict_total.copy()

        for code in self.list_code_selected:
            if code in dict_data:
                x, y, vol = dict_data[code]
                trader: Trader = self.dict_trader[code]
                trader.setTradeData(x, y, vol)

                # 銘柄単位の現在株価および含み益と収益を更新
                trader.dock.setPrice(y)
                trader.dock.setProfit(dict_profit[code])
                trader.dock.setTotal(dict_total[code])

    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
    # 取引ボタンがクリックされた時の処理（リアルタイム用）
    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
    def on_buy(self, code: str, price: float, note: str):
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 買建で建玉取得リクエストのシグナル
        self.requestPositionOpen.emit(
            code, self.ts_system, price, ActionType.BUY, note
        )
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def on_sell(self, code: str, price: float, note: str):
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 売建で建玉取得リクエストのシグナル
        self.requestPositionOpen.emit(
            code, self.ts_system, price, ActionType.SELL, note
        )
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def on_repay(self, code: str, price: float, note: str):
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 建玉返済リクエストのシグナル
        self.requestPositionClose.emit(
            code, self.ts_system, price, note
        )
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    ###########################################################################
    #
    # デバッグ（レビュー）用メソッド
    #
    ###########################################################################
    def on_create_thread_review(self, excel_path: str, list_code_selected: list):
        """
        レビュー用ティックデータ取得スレッドの生成
        :param excel_path:
        :param list_code_selected:
        :return:
        """
        self.list_code_selected = list_code_selected

        # ザラ場の開始時間などのタイムスタンプ取得（Excelの日付）
        self.dict_ts = get_intraday_timestamp(excel_path)
        # ---------------------------------------------------------------------
        # 00. デバッグ/レビュー用データ取得インスタンスの生成
        self.worker = worker = ExcelReviewWorker(excel_path)
        worker.moveToThread(self.thread)
        # ---------------------------------------------------------------------
        # 01. データ読み込み済みの通知（レビュー用のみ）
        worker.notifyDataReady.connect(self.set_data_ready_status)
        # =====================================================================
        # 02. スレッドが開始されたら、ワーカースレッド内で初期化処理を実行するシグナルを発行
        self.thread.started.connect(self.requestWorkerInit.emit)
        # ---------------------------------------------------------------------
        # 03. 初期化処理は指定された Excel ファイルの読み込み
        self.requestWorkerInit.connect(worker.initWorker)
        # ---------------------------------------------------------------------
        # 04. 売買ポジション処理用のメソッドへキューイング
        self.requestPositionOpen.connect(worker.posman.openPosition)
        self.requestPositionClose.connect(worker.posman.closePosition)
        # ---------------------------------------------------------------------
        # 05. 取引結果を取得するメソッドへキューイング
        self.requestTransactionResult.connect(worker.getTransactionResult)
        # ---------------------------------------------------------------------
        # 06. 現在株価を取得するメソッドへキューイング。
        self.requestCurrentPrice.connect(worker.readCurrentPrice)
        # ---------------------------------------------------------------------
        # 07. データフレームを保存するメソッドへキューイング
        # デバッグ/レビュー用では本機能なし
        # ---------------------------------------------------------------------
        # 08. スレッドを終了する下記のメソッドへキューイング（リアルタイムでは xlwings 関連）。
        self.requestStopProcess.connect(worker.stopProcess)
        # =====================================================================
        # 10. 初期化後の銘柄情報を通知
        worker.notifyTickerN.connect(self.on_create_trader)
        # ---------------------------------------------------------------------
        # 11. タイマーで現在時刻と株価を通知
        worker.notifyCurrentPrice.connect(self.on_update_data)
        # ---------------------------------------------------------------------
        # 12. 取引結果を通知
        worker.notifyTransactionResult.connect(self.on_transaction_result)
        # ---------------------------------------------------------------------
        # 13. データフレームを保存終了を通知
        # デバッグ/レビュー用では本機能なし
        # ---------------------------------------------------------------------
        # 19. スレッド終了関連
        worker.threadFinished.connect(self.on_thread_finished)
        # =====================================================================
        # 20. スレッドを開始
        self.thread.start()

    def on_request_data_review(self):
        """
        タイマー処理（デバッグ/レビュー用）
        """
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 現在価格の取得要求をワーカースレッドに通知
        self.requestCurrentPrice.emit(self.ts_system)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # システム時間のインクリメント（１秒）
        self.ts_system += 1

        # 取引時間を過ぎたかをチェック
        if self.dict_ts["end_2h"] < self.ts_system <= self.dict_ts["ca"]:
            if not self.finished_trading:
                # ポジションがあればクローズする
                self.force_closing_position()
                # このフラグにより、何回もポジションがあるかどうかの確認を繰り返さない。
                self.finished_trading = True
        elif self.dict_ts["end"] < self.ts_system:
            self.timer.stop()
            self.logger.info(f"{__name__}: タイマーを停止しました。")
            # 取引結果を取得
            self.requestTransactionResult.emit()

        # ツールバーの時刻を更新（現在時刻を表示するだけ）
        self.toolbar.updateTime(self.ts_system)

    def on_review_play(self):
        """
        読み込んだデータ・レビュー開始（デバッグ/レビュー用）
        :return:
        """
        if self.flag_data_ready:
            self.ts_system = self.dict_ts["start"]
            # タイマー開始
            self.timer.start()
            self.logger.info(f"{__name__}: タイマーを開始しました。")

    def on_review_stop(self):
        """
        読み込んだデータ・レビュー停止（デバッグ/レビュー用）
        :return:
        """
        if self.timer.isActive():
            self.timer.stop()
            self.logger.info(f"{__name__}: タイマーを停止しました。")
            # 取引結果を取得
            self.requestTransactionResult.emit()

    def set_data_ready_status(self, state: bool):
        self.flag_data_ready = state
        self.logger.info(
            f"{__name__}: データ準備完了フラグが {state} になりました。"
        )
