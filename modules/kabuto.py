import logging
import os
import time

import pandas as pd
from PySide6.QtCore import (
    QThread,
    QTimer,
    Qt,
    Signal,
    Slot,
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
from funcs.tide import conv_date_str_to_path, get_intraday_timestamp
from funcs.tse import get_ticker_name_list
from funcs.uis import clear_boxlayout
from modules.reviewer import ExcelReviewWorker
from modules.rssreader import RSSReaderWorker
from widgets.dialogs import DlgAboutThis, DlgCodeSel
from modules.dlg_system_settings import DlgSystemSettings
from widgets.misc import TickFont
from widgets.statusbars import StatusBar
from modules.toolbar import ToolBar
from modules.trader import Trader
from modules.win_transaction import WinTransaction
from structs.res import AppRes
from widgets.containers import ScrollArea, Widget
from widgets.layouts import VBoxLayout


class Kabuto(QMainWindow):
    __app_name__ = "Kabuto"
    __version__ = "0.5.0"
    __author__ = "Fuhito Suguri"
    __license__ = "MIT"

    # インスタンス変数の型宣言
    logger: logging.Logger
    res: AppRes
    timer_interval: int
    flag_data_ready: bool
    thread: QThread | None
    worker: ExcelReviewWorker | RSSReaderWorker | None
    trader: Trader | None
    dict_trader: dict[str, Trader]
    list_code: list[str]
    list_code_selected: list[str]
    df_transaction: pd.DataFrame | None
    win_transaction: WinTransaction | None
    ts_system: float
    dict_ts: dict[str, float | str]
    finished_trading: bool
    toolbar: ToolBar
    statusbar: StatusBar
    area_chart: ScrollArea
    layout: VBoxLayout
    timer: QTimer
    font_tick: TickFont

    # ワーカーの初期化シグナル
    requestWorkerInit = Signal()

    # 現在価格取得リクエスト・シグナル
    requestCurrentPrice = Signal(float)
    requestSaveDataFrame = Signal()
    requestStopProcess = Signal()

    # 売買
    requestBuy = Signal(str, float, float, str)
    requestSell = Signal(str, float, float, str)
    requestRepay = Signal(str, float, float, str)
    requestTransactionResult = Signal()

    # このスレッドが開始されたことを通知するシグナル（デバッグ用など）
    threadReady = Signal()

    def __init__(self, debug: bool = True) -> None:
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.res = AppRes()
        self.res.debug = debug

        # リアルタイム／デバッグモード設定
        self._init_mode_settings()

        # データ構造初期化（スレッド関連）
        self._init_data_structures()

        # UI セットアップ
        self._setup_ui()

        # タイマー初期化
        self._init_timer()

    def _init_mode_settings(self) -> None:
        """モード別設定の初期化"""
        if self.res.debug:
            self.logger.info(f"デバッグモードで起動しました。")
            self.timer_interval = 100
            self.flag_data_ready = False
        else:
            self.logger.info(f"通常モードで起動しました。")
            self.timer_interval = 2000

    def _init_data_structures(self) -> None:
        """データ構造の初期化"""
        # スレッド/ワーカー
        self.thread = QThread(self)
        self.worker = None

        # Trader 関連
        self.trader = None
        self.dict_trader = {}
        self.list_code = []
        self.list_code_selected = []

        # 取引履歴
        self.df_transaction = None
        self.win_transaction = None

        # 時刻関連
        self.ts_system = 0.0
        self.dict_ts = get_intraday_timestamp()
        self.finished_trading = False

    def _init_timer(self) -> None:
        """タイマーの初期化とシグナル接続"""
        self.timer = timer = QTimer()
        timer.setInterval(self.timer_interval)
        if self.res.debug:
            # デバッグモードではティックデータを読み込んでからスレッドを起動
            timer.timeout.connect(self.on_request_data_review)
        else:
            # リアルタイムモードでは、直ちにスレッドを起動
            timer.timeout.connect(self.on_request_data)
            # RSS 用 Excel ファイルを指定して xlwings を利用するスレッド
            self.on_create_thread()

    def _setup_ui(self) -> None:
        """UI コンポーネントの初期化"""
        # アプリケーション・フォント
        self.font_tick = TickFont(self.res.path_monospace)
        self.res.name_tick_font = self.font_tick.name
        # ウィンドウアイコンとタイトルを設定
        self.setWindowIcon(QIcon(os.path.join(self.res.dir_image, "kabuto.png")))
        title_win = f"{self.__app_name__} - {self.__version__}"
        if self.res.debug:
            # デバッグモードを示す文字列を追加
            title_win = f"{title_win} [debug mode]"
        self.setWindowTitle(title_win)
        # ---------------------------------------------------------------------
        # ツールバー
        # ---------------------------------------------------------------------
        self.toolbar = toolbar = ToolBar(self.res)
        toolbar.clickedAbout.connect(self.on_about)
        toolbar.clickedPlay.connect(self.on_review_play)
        toolbar.clickedStop.connect(self.on_review_stop)
        toolbar.clickedSetting.connect(self.on_setting)
        toolbar.clickedTransaction.connect(self.on_show_transaction)
        toolbar.requestSwitchCharts.connect(self.on_switch_charts)
        toolbar.selectedExcelFile.connect(self.on_create_thread_review)
        self.addToolBar(toolbar)
        # ---------------------------------------------------------------------
        # ステータスバー
        # ---------------------------------------------------------------------
        self.statusbar = statusbar = StatusBar(self.res)
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

    def _connect_worker_signals(
            self,
            worker: ExcelReviewWorker | RSSReaderWorker
    ) -> None:
        """ワーカーとのシグナル接続（共通処理）"""
        # データ読み込み済みの通知
        worker.notifyDataReady.connect(self.set_data_ready_status)

        # スレッド開始時の初期化
        self.thread.started.connect(self.requestWorkerInit.emit)  # type: ignore
        self.requestWorkerInit.connect(worker.initWorker)

        # 売買処理
        self.requestBuy.connect(worker.macro_do_buy)
        self.requestSell.connect(worker.macro_do_sell)
        self.requestRepay.connect(worker.macro_do_repay)

        # 取引結果・現在価格
        self.requestTransactionResult.connect(worker.getTransactionResult)
        self.requestCurrentPrice.connect(worker.readCurrentPrice)

        # データフレーム保存
        self.requestSaveDataFrame.connect(worker.saveDataFrame)

        # スレッド終了
        self.requestStopProcess.connect(worker.stopProcess)

        # 通知受信
        worker.notifyTickerN.connect(self.on_create_trader)
        worker.notifyCurrentPrice.connect(self.on_update_data)
        worker.notifyTransactionResult.connect(self.on_transaction_result)
        worker.saveCompleted.connect(self.on_save_completed)
        worker.sendResult.connect(self.order_execution_result)
        worker.threadFinished.connect(self.on_thread_finished)

    def closeEvent(self, event: QCloseEvent) -> None:
        """アプリ終了イベント"""
        self._stop_timer()
        self._cleanup_traders()
        self._cleanup_thread()
        self.logger.info(f"停止して閉じました。")
        event.accept()

    def _stop_timer(self) -> None:
        """タイマーの停止"""
        if self.timer.isActive():
            self.timer.stop()
            self.logger.info(f"タイマーを停止しました。")

    def _cleanup_traders(self) -> None:
        """Trader インスタンスのクリーンアップ"""
        if self.dict_trader:
            self.logger.info(f"Trader インスタンスの終了処理を開始します。")
            for code, trader in self.dict_trader.items():
                try:
                    # Trader の終了処理を呼び出す
                    if trader is not None and trader.thread.isRunning():
                        self.logger.info(f"Trader ({code}) のスレッドを終了します。")

                        # ワーカーにクリーンアップを実行させる
                        trader.requestCleanup.emit()

                        # 少し待ってクリーンアップが完了するのを待つ
                        QThread.msleep(100)

                        # スレッドに終了を要求
                        trader.thread.quit()

                        # タイムアウト付きで待機（5秒）
                        if not trader.thread.wait(5000):
                            self.logger.warning(
                                f"Trader ({code}) のスレッドが応答しません。強制終了します。"
                            )
                            trader.thread.terminate()
                            trader.thread.wait(1000)

                        self.logger.info(f"Trader ({code}) のスレッドを終了しました。")
                except Exception as e:
                    self.logger.error(f"Trader ({code}) の終了処理でエラー: {e}")

            # Trader 辞書をクリア
            self.dict_trader.clear()
            self.logger.info(f"すべての Trader インスタンスを終了しました。")

    def _cleanup_thread(self) -> None:
        """スレッドとワーカーのクリーンアップ"""
        try:
            if self.thread.isRunning():
                self.requestStopProcess.emit()
                time.sleep(1)

            if self.thread is not None:
                self.thread.quit()
                self.thread.wait()
                self.logger.info(f"スレッド self.thread を削除しました。")

            if self.worker is not None:
                self.worker.deleteLater()
                self.worker = None
                self.logger.info(f"ワーカー self.worker を削除しました。")

            if self.thread is not None:
                self.thread.deleteLater()
                self.thread = None
        except RuntimeError as e:
            self.logger.error(f"終了時にエラー発生: {e}")

    def create_trader(self, dict_name: dict[str, str]) -> None:
        """
        選択した銘柄数分の Trader インスタンスの生成
        （リアルタイム・モード、デバッグ・モード共通）
        :param dict_name:
        :return:
        """
        # 配置済みの Trader インスタンスを消去
        clear_boxlayout(self.layout)
        # Trader 辞書のクリア
        self.dict_trader.clear()
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

    def force_closing_position(self) -> None:
        """
        強制的にポジションを解消
        :return:
        """
        self.logger.info(f"売買を強制終了します。")
        for code in self.dict_trader.keys():
            trader: Trader = self.dict_trader[code]
            trader.forceRepay()

    def on_about(self) -> None:
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

    def on_create_thread(self) -> None:
        """リアルタイム用スレッドの生成"""
        self.worker = RSSReaderWorker(self.res)
        self.worker.moveToThread(self.thread)
        self._connect_worker_signals(self.worker)
        self.thread.start()

    # def on_create_trader(self, list_code: list, dict_name: dict):
    def on_create_trader(
            self,
            list_code: list[str],
            dict_name: dict[str, str]
    ) -> None:
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
            self.logger.info(f"レビューの準備ができました。")
            return

        # ---------------------------------------------------------------------
        # Excel から読み取った銘柄を標準出力（確認用）
        # ---------------------------------------------------------------------
        self.logger.info(f"ティックデータ収集銘柄一覧")
        for code in list_code:
            self.logger.info(f"{code}, {dict_name[code]}")
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
            self.logger.info(f"タイマーを開始しました。")

    def on_request_data(self) -> None:
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
            self.logger.info(f"タイマーを停止しました。")
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # 🧿 取引結果を取得
            self.requestTransactionResult.emit()
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # 🧿 収集したデータの保存
            self.requestSaveDataFrame.emit()
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # 保持したテクニカルデータを保存
            if self.toolbar.isAlt():
                self.logger.info(
                    f"代替環境なのでテクニカルデータの保存をキャンセルします。"
                )
            else:
                """
                バックアップ用に稼働しているのでなければ、テクニカルデータを保存
                ※ このデータは Github にアップしているので上書きや衝突を防ぐため
                """
                path_dir = os.path.join(
                    self.res.dir_output,
                    conv_date_str_to_path(self.dict_ts["datetime_str"])
                )
                self.save_technicals(path_dir)
        else:
            pass

        # ツールバーの時刻を更新
        self.toolbar.updateTime(self.ts_system)

    def on_save_completed(self, state: bool) -> None:
        if state:
            self.logger.info(f"ティック・データを正常に保存しました。")
        else:
            self.logger.info(f"ティック・データを正常に保存できませんでした。")

    def on_setting(self):
        dialog = DlgSystemSettings(self.res)
        result = dialog.exec()
        if result == QDialog.DialogCode.Accepted:
            print("OKが押されました")
        elif result == QDialog.DialogCode.Rejected:
            print("キャンセルされました")

    def on_show_transaction(self) -> None:
        """
        取引明細の表示
        :return:
        """
        self.win_transaction = WinTransaction(self.res, self.df_transaction)
        self.win_transaction.show()

    def on_switch_charts(self, state: bool):
        for code in self.dict_trader.keys():
            trader: Trader = self.dict_trader[code]
            trader.switchChartType(state)

    def on_thread_finished(self, result: bool) -> None:
        """
        スレッド終了時のログ
        :param result:
        :return:
        """
        if result:
            self.logger.info(f"スレッドが正常終了しました。")
        else:
            self.logger.error(f"スレッドが異常終了しました。")
        # タイマーの停止
        if self.timer.isActive():
            self.timer.stop()
            self.logger.info(f"タイマーを停止しました。")

    def on_ticker_ready(self, code: str) -> None:
        self.logger.info(f"銘柄コード {code} のスレッドの準備ができました。")

    def on_transaction_result(self, df: pd.DataFrame) -> None:
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
        self.logger.info(f"取引明細を {path_trans} に保存しました。")
        # インスタンス変数に取引明細を保持
        self.df_transaction = df
        # ツールバーの「取引履歴」ボタンを Enabled にする
        self.toolbar.set_transaction()

    def on_update_data(
            self,
            dict_data: dict[str, tuple[float, float, float]],
            dict_profit: dict[str, float],
            dict_total: dict[str, float]
    ) -> None:
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
                profit = dict_profit[code]
                total = dict_total[code]
                trader: Trader = self.dict_trader[code]
                trader.setTradeData(x, y, vol, profit, total)

    def save_technicals(self, path_dir: str) -> None:
        """
        取引終了後に銘柄毎にテクニカルデータを保存
        :param path_dir:
        :return:
        """
        os.makedirs(path_dir, exist_ok=True)
        for code in self.list_code_selected:
            trader: Trader = self.dict_trader[code]
            trader.saveTechnicals(path_dir)

    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
    # 取引ボタンがクリックされた時の処理
    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
    def on_buy(self, code: str, price: float, note: str) -> None:
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 買建リクエストのシグナル
        self.requestBuy.emit(code, self.ts_system, price, note)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def on_sell(self, code: str, price: float, note: str) -> None:
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 売建リクエストのシグナル
        self.requestSell.emit(code, self.ts_system, price, note)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def on_repay(self, code: str, price: float, note: str) -> None:
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 建玉返済リクエストのシグナル
        self.requestRepay.emit(code, self.ts_system, price, note)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    @Slot(str, bool)
    def order_execution_result(self, code: str, price: float) -> None:
        """
        約定確認結果
        :param code:
        :param status:
        :return:
        """
        trader: Trader = self.dict_trader[code]
        # 発注確認結果
        trader.orderExecResult(price)

    ###########################################################################
    #
    # デバッグ（レビュー）用メソッド
    #
    ###########################################################################
    def on_create_thread_review(
            self,
            excel_path: str,
            list_code_selected: list[str]
    ) -> None:
        """デバッグ用スレッドの生成"""
        self.list_code_selected = list_code_selected
        self.dict_ts = get_intraday_timestamp(excel_path)

        self.worker = ExcelReviewWorker(excel_path)
        self.worker.moveToThread(self.thread)
        self._connect_worker_signals(self.worker)  # 引数で渡す
        self.thread.start()

    def on_request_data_review(self) -> None:
        """
        タイマー処理（デバッグ/レビュー用）
        """
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 現在価格の取得要求をワーカースレッドに通知
        self.requestCurrentPrice.emit(self.ts_system)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # システム時間のインクリメント（１秒）
        self.ts_system += 1.0

        # 取引時間を過ぎたかをチェック
        if self.dict_ts["end_2h"] < self.ts_system <= self.dict_ts["ca"]:
            if not self.finished_trading:
                # ポジションがあればクローズする
                self.force_closing_position()
                # このフラグにより、何回もポジションがあるかどうかの確認を繰り返さない。
                self.finished_trading = True
        elif self.dict_ts["end"] < self.ts_system:
            self.timer.stop()
            self.logger.info(f"タイマーを停止しました。")
            # 取引結果を取得
            self.requestTransactionResult.emit()
            # 保持したテクニカルデータを保存
            path_dir = os.path.join(
                self.res.dir_temp,
                conv_date_str_to_path(self.dict_ts["datetime_str"])
            )
            self.save_technicals(path_dir)

        # ツールバーの時刻を更新（現在時刻を表示するだけ）
        self.toolbar.updateTime(self.ts_system)

    def on_review_play(self) -> None:
        """
        読み込んだデータ・レビュー開始（デバッグ/レビュー用）
        :return:
        """
        if self.flag_data_ready:
            self.ts_system = self.dict_ts["start"]
            # タイマー開始
            self.timer.start()
            self.logger.info(f"タイマーを開始しました。")

    def on_review_stop(self) -> None:
        """
        読み込んだデータ・レビュー停止（デバッグ/レビュー用）
        :return:
        """
        if self.timer.isActive():
            self.timer.stop()
            self.logger.info(f"タイマーを停止しました。")
            # 取引結果を取得
            self.requestTransactionResult.emit()

            # 保持したテクニカルデータを保存
            path_dir = os.path.join(
                self.res.dir_temp,
                conv_date_str_to_path(self.dict_ts["datetime_str"])
            )
            self.save_technicals(path_dir)

    def set_data_ready_status(self, state: bool) -> None:
        self.flag_data_ready = state
        self.logger.info(
            f"データ準備完了フラグが {state} になりました。"
        )
        # Play / Stop ボタンの状態変更
        self.toolbar.switch_playstop(state)
