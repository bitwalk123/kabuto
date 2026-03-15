import logging
import os
from typing import Any, Literal, TypeAlias

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QMainWindow, QDialog

from funcs.setting import load_setting
from modules.agent import WorkerAgent
from modules.dock import DockTrader
from modules.trend_charts import TrendCharts
from structs.app_enum import (
    ActionType,
    PositionType,
)
from structs.res import AppRes
from modules.chart import TrendChart
from widgets.dialogs import DlgSetting

# 型エイリアスの定義（クラスの外に配置）
TradeAction: TypeAlias = Literal["doBuy", "doSell", "doRepay"]
TradeKey: TypeAlias = tuple[ActionType, PositionType]


class Trader(QMainWindow):
    # 環境クラス用
    sendTradeData = Signal(float, float, float)
    requestResetEnv = Signal()
    requestSaveTechnicals = Signal(str)

    # 売買用
    requestPositionOpen = Signal(ActionType)
    requestPositionClose = Signal()
    requestTransactionResult = Signal()

    # クリーンアップ要求用シグナル
    requestCleanup = Signal()

    # --- 状態遷移表 ---
    ACTION_DISPATCH: dict[TradeKey, TradeAction] = {
        (ActionType.BUY, PositionType.NONE): "doBuy",  # 建玉がなければ買建
        (ActionType.BUY, PositionType.SHORT): "doRepay",  # 売建（ショート）であれば（買って）返済
        (ActionType.SELL, PositionType.NONE): "doSell",  # 建玉がなければ売建
        (ActionType.SELL, PositionType.LONG): "doRepay",  # 買建（ロング）であれば（売って）返済
        # HOLD は何もしないので載せない
    }

    def __init__(self, res: AppRes, code: str, dict_ts: dict[str, Any]) -> None:
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.res = res
        self.code = code
        self.dict_ts = dict_ts

        # ティックデータ
        self.ts = 0
        self.price = 0

        # テクニカル指標
        # self.vwap: float = 0.0
        self.list_ts: list[float] = []
        self.list_vwap: list[float] = []
        self.list_ma_1: list[float] = []
        self.list_rsi: list[float] = []

        self.dict_trend = {
            "ts": self.list_ts,
            "ma_1": self.list_ma_1,
            "vwap": self.list_vwap,
            "rsi": self.list_rsi,
        }

        # 銘柄コード別設定ファイルの取得
        self.dict_setting: dict[str, Any] = load_setting(res, code)

        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        #  UI
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_

        # ---------------------------------------------------------------------
        # 右側のドック
        # ---------------------------------------------------------------------
        self.dock = dock = DockTrader(res, code)
        self.dock.clickedBuy.connect(self.on_buy)
        self.dock.clickedSell.connect(self.on_sell)
        self.dock.clickedRepay.connect(self.on_repay)
        self.dock.clickedSetting.connect(self.on_setting)
        self.dock.clickedSave.connect(self.on_save)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

        # ---------------------------------------------------------------------
        # チャート・インスタンス
        # ---------------------------------------------------------------------
        # self.trend = trend = TrendChart(res, dict_ts, self.dict_setting)
        self.trend = trend = TrendCharts(res, dict_ts, self.dict_setting)
        self.setCentralWidget(trend)

        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # 売買モデル用スレッド
        self.thread = QThread(self)

        # 学習済みモデルのパス
        # path_model = get_trained_ppo_model_path(res, code)

        # ワーカースレッドの生成
        self.worker = worker = WorkerAgent(code, self.dict_setting)
        worker.moveToThread(self.thread)

        # メインスレッドのシグナル処理 → ワーカースレッドのスロットへ
        self.requestResetEnv.connect(worker.resetEnv)
        self.sendTradeData.connect(worker.addData)
        self.requestSaveTechnicals.connect(worker.saveTechnicals)
        self.requestPositionOpen.connect(worker.env.openPosition)
        self.requestPositionClose.connect(worker.env.closePosition)

        # ワーカースレッドからのシグナル処理 → メインスレッドのスロットへ
        worker.completedResetEnv.connect(self.reset_env_completed)
        worker.completedTrading.connect(self.on_trading_completed)
        worker.notifyAction.connect(self.on_action)
        worker.sendTechnicals.connect(self.on_technicals)

        # クリーンアップシグナルを接続
        self.requestCleanup.connect(self.worker.cleanup)

        # スレッド終了時にワーカーを自動削除
        self.thread.finished.connect(self.worker.deleteLater)

        # スレッドの開始
        self.thread.start()
        # エージェント環境のリセット → リセット終了で処理開始
        self.requestResetEnv.emit()
        #
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_

    def closeEvent(self, event: QCloseEvent) -> None:
        """
        ウィンドウを閉じる際のクリーンアップ処理
        """
        if self.thread.isRunning():
            self.logger.info(f"スレッドの終了を開始します。")

            # ワーカーにクリーンアップを実行させる
            self.requestCleanup.emit()

            # 少し待ってクリーンアップが完了するのを待つ
            QThread.msleep(100)

            # スレッドに終了を要求
            self.thread.quit()

            # タイムアウト付きで待機（5秒）
            if not self.thread.wait(5000):
                self.logger.warning(f"スレッドが5秒以内に応答しませんでした。強制終了します。")
                self.thread.terminate()
                self.thread.wait(1000)

            self.logger.info(f"スレッドを終了しました。")

        event.accept()

    def forceRepay(self) -> None:
        """
        強制的に建玉返済
        :return:
        """
        self.dock.force_repay()

    def on_action(self, action: int, position: PositionType) -> None:
        """
        売買アクション
        :param action:
        :param position:
        :return:
        """
        action_enum = ActionType(action)

        # HOLD は即 return
        if action_enum == ActionType.HOLD:
            return

        method_name = self.ACTION_DISPATCH.get((action_enum, position))
        if method_name is None:
            self.logger.error(
                f"trade rule violation! action={action_enum}, pos={position}"
            )
            return

        # dock のメソッドを取得して実行
        getattr(self.dock, method_name)()

    def on_save(self) -> None:
        """
        チャートを保存
        :return:
        """
        # 保存先のパス
        file_img = f"{self.code}_trend.png"
        if self.res.debug:
            output_dir: str = os.path.join(
                self.res.dir_temp,
                self.dict_ts['datetime_str_3']
            )
        else:
            output_dir: str = os.path.join(
                self.res.dir_output,
                self.dict_ts['datetime_str_3']
            )

        # パスの階層がなかったら生成して保存
        os.makedirs(output_dir, exist_ok=True)
        path_img = os.path.join(output_dir, file_img)
        self.trend.save(path_img)

    def on_setting(self):
        dialog = DlgSetting(self.res, self.code, self.dict_setting)
        result = dialog.exec()
        if result == QDialog.DialogCode.Accepted:
            print("OKが押されました")
        elif result == QDialog.DialogCode.Rejected:
            print("キャンセルされました")

    def on_technicals(self, dict_technicals: dict[str, Any]) -> None:
        if dict_technicals["warmup"]:
            self.dock.panel_trading.lockButtons()
        else:
            self.dock.panel_trading.unLockButtons()

        # テクニカル指標
        self.list_ts.append(dict_technicals["ts"])
        # self.vwap = dict_technicals["vwap"]
        self.list_vwap.append(dict_technicals["vwap"])
        self.list_ma_1.append(dict_technicals["ma1"])
        self.list_rsi.append(dict_technicals["rsi"])

        # クロス時の縦線表示 1
        if 0.0 < dict_technicals["cross1"]:
            self.trend.setCrossGolden(dict_technicals["ts"])
        elif dict_technicals["cross1"] < 0.0:
            self.trend.setCrossDead(dict_technicals["ts"])

        # クロス時の縦線表示 2
        if 0.0 < dict_technicals["cross2"]:
            self.trend.setCrossGolden(dict_technicals["ts"])
        elif dict_technicals["cross2"] < 0.0:
            self.trend.setCrossDead(dict_technicals["ts"])

        self.update_technicals()

    def on_trading_completed(self) -> None:
        self.logger.info("取引が終了しました。")

    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
    # 取引ボタンがクリックされた時の処理
    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
    def on_buy(self, code: str, price: float, note: str, auto: bool) -> None:
        if not auto:
            # Agent からの売買要求で返ってきた売買シグナルを Agent に戻さない
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # 🧿 買建で建玉取得リクエストのシグナル
            self.requestPositionOpen.emit(ActionType.BUY)
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def on_sell(self, code: str, price: float, note: str, auto: bool) -> None:
        if not auto:
            # Agent からの売買要求で返ってきた売買シグナルを Agent に再び戻さない
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # 🧿 売建で建玉取得リクエストのシグナル
            self.requestPositionOpen.emit(ActionType.SELL)
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def on_repay(self, code: str, price: float, note: str, auto: bool) -> None:
        if not auto:
            # Agent からの売買要求で返ってきた売買シグナルを Agent に再び戻さない
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # 🧿 建玉返済リクエストのシグナル
            self.requestPositionClose.emit()
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def reset_env_completed(self) -> None:
        """
        環境をリセット済
        :return:
        """
        msg = f"銘柄コード {self.code} 用の環境がリセットされました。"
        self.logger.info(msg)

    def saveTechnicals(self, path_dir: str) -> None:
        """
        保持したテクニカル指標のデータを指定パスに保存
        :param path_dir:
        :return:
        """
        path_csv = os.path.join(path_dir, f"{self.code}_technicals.csv")
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 テクニカルデータ保存リクエストのシグナル
        self.requestSaveTechnicals.emit(path_csv)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def orderExecResult(self, price: float) -> None:
        """
        発注結果
        :param price:
        :return:
        """
        # 売買返済ボタンのロックを解除、次の状態設定
        if self.dock.next_trading_buttons_status(price):
            self.trend.setEvenLine(price)
        else:
            self.trend.setEvenLine(0.0)

    def setChartTitle(self, title: str) -> None:
        """
        チャートのタイトルを設定
        :param title:
        :return:
        """
        self.trend.setTrendTitle(title)

    def setTimeAxisRange(self, ts_start: float, ts_end: float) -> None:
        """
        x軸のレンジ
        固定レンジで使いたいため。
        ただし、前場と後場で分ける機能を検討する余地はアリ
        :param ts_start:
        :param ts_end:
        :return:
        """
        self.trend.setXRange(ts_start, ts_end)

    def setTradeData(
            self,
            ts: float,
            price: float,
            volume: float,
            profit: float,
            total: float
    ) -> None:
        """
        株価データなどをセット
        :param ts:
        :param price:
        :param volume:
        :param profit:
        :param total:
        :return:
        """

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # ティックデータを送るシグナル
        self.sendTradeData.emit(ts, price, volume)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        self.ts = ts
        self.price = price

        # 株価トレンド線
        self.trend.setDot([ts], [price])

        # 銘柄単位の現在株価および含み益と収益を更新
        self.dock.setPrice(price)
        self.dock.setProfit(profit)
        self.dock.setTotal(total)

    def update_technicals(self) -> None:
        """
        if flag:
        disparity line
           self.trend.setTechnicals(self.dict_disparity, False)
        else:
        """
        self.trend.setTechnicals(self.dict_trend)
