import logging
import os
import sys

import pandas as pd
from PySide6.QtCore import QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QMainWindow

from rhino.rhino_funcs import get_intraday_timestamp
from rhino.rhino_review import RhinoReview
from rhino.rhino_toolbar import RhinoToolBar
from structs.res import AppRes
from widgets.containers import Widget
from widgets.layouts import VBoxLayout

if sys.platform == "win32":
    debug = False
else:
    debug = True  # Windows 以外はデバッグ・モード


class Rhino(QMainWindow):
    __app_name__ = "Rhino"
    __version__ = "0.9.0"
    __author__ = "Fuhito Suguri"
    __license__ = "MIT"

    def __init__(self, options: list = None):
        super().__init__()
        global debug  # グローバル変数であることを明示
        self.res = res = AppRes()
        self.logger = logging.getLogger(__name__)  # モジュール固有のロガーを取得

        # コンソールから起動した際のオプションをチェック
        if len(options) > 0:
            for option in options:
                if option == "debug":
                    debug = True  # Windows 上でデバッグ・モードを使用する場合
        # デバッグ・モードを保持
        res.debug = debug

        #######################################################################
        # NORMAL / DEBUG モード固有の設定
        if debug:
            # DEBUG モード
            self.logger.info(f"{__name__} executed as DEBUG mode!")
            self.timer_interval = 100  # タイマー間隔（ミリ秒）（デバッグ時）
        else:
            # NORMAL モード
            self.logger.info(f"{__name__} executed as NORMAL mode!")
            self.timer_interval = 1000  # タイマー間隔（ミリ秒）
        #
        #######################################################################

        # スレッド用インスタンス
        self.review: RhinoReview | None = None

        # ザラ場の開始時間などのタイムスタンプ取得（本日分）
        self.dict_ts = get_intraday_timestamp()

        # ---------------------------------------------------------------------
        #  UI
        # ---------------------------------------------------------------------
        # ウィンドウアイコンとタイトルを設定
        icon = QIcon(os.path.join(res.dir_image, "rhino.png"))
        self.setWindowIcon(icon)
        self.setWindowTitle(f"{self.__app_name__} - {self.__version__}")

        # ツールバー
        self.toolbar = toolbar = RhinoToolBar(res)
        toolbar.excelSelected.connect(self.on_create_review_thread)
        self.addToolBar(toolbar)

        # メインウィジェット
        base = Widget()
        self.setCentralWidget(base)
        self.layout = layout = VBoxLayout()
        base.setLayout(layout)

        # ---------------------------------------------------------------------
        # タイマー
        # ---------------------------------------------------------------------
        self.timer = timer = QTimer()
        timer.setInterval(self.timer_interval)

    def on_thread_finished(self, result: bool):
        """
        スレッド終了時のログ
        :param result:
        :return:
        """
        if result:
            self.logger.info("スレッドが正常終了しました。")
        else:
            self.logger.error("スレッドが異常終了しました。")

        if self.timer.isActive():
            self.timer.stop()
            self.logger.info("タイマーを停止しました。")

    def on_transaction_result(self, df: pd.DataFrame):
        """
        取引結果のデータフレームを取得（リアルタイム、デバッグ・モード共通）
        :param df:
        :return:
        """
        print(df)
        print("合計損益", df["損益"].sum())

        # インスタンス変数に保存
        self.df_transaction = df

        # ツールバーの「取引履歴」ボタンを Enabled にする
        self.toolbar.set_transaction()

    def on_update_data(self, dict_data, dict_profit, dict_total):
        """
        ティックデータ、含み益、損益の更新
        :param dict_data:
        :param dict_profit:
        :param dict_total:
        :return:
        """
        for ticker in dict_data.keys():
            x, y = dict_data[ticker]
            trader = self.dict_trader[ticker]
            # trader.setTimePrice(x, y)
            trader.dock.setPrice(y)
            # 銘柄単位の含み益と収益を更新
            trader.dock.setProfit(dict_profit[ticker])
            trader.dock.setTotal(dict_total[ticker])
            # Parabolic SAR
            thread_ticker: ThreadTicker = self.dict_thread_ticker[ticker]
            # ここで PSAR を算出する処理が呼び出される
            thread_ticker.notifyNewPrice.emit(x, y)

    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
    # デバッグ用メソッド
    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
    def on_create_review_thread(self, excel_path: str):
        self.review = RhinoReview(excel_path)
        # 初期化後の銘柄情報を通知
        self.review.worker.notifyTickerN.connect(self.on_create_trader_review)
        # タイマーで現在時刻と株価を通知
        self.review.worker.notifyCurrentPrice.connect(self.on_update_data)
        # 取引結果を通知
        self.review.worker.notifyTransactionResult.connect(self.on_transaction_result)
        # スレッド終了関連
        self.review.worker.threadFinished.connect(self.on_thread_finished)

    def on_create_trader_review(self, list_ticker: list, dict_name: dict, dict_lastclose: dict):
        """
        Trader インスタンスの生成（デバッグ用）
        :param list_ticker:
        :param dict_name:
        :param dict_lastclose:
        :return:
        """
        # 銘柄数分の Trader インスタンスの生成
        self.create_trader(list_ticker, dict_name, dict_lastclose)

        # デバッグの場合はスタート・ボタンがクリックされるまでは待機
        # self.data_ready = True
        self.logger.info("レビューの準備完了です。")
