import logging
import os
from typing import Any, Literal, TypeAlias

import pandas as pd
from PySide6.QtCore import (
    Qt,
    QThread,
    Signal,
)
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QMainWindow

from funcs.setting import load_setting
from modules.agent import WorkerAgent
from modules.dock import DockTrader
from structs.app_enum import ActionType, PositionType
from structs.res import AppRes
from modules.chart import TrendChart

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
        self.list_x: list[float] = []
        self.list_y: list[float] = []
        self.list_v: list[float] = []

        # テクニカル指標
        self.vwap: float = 0.0
        self.list_ts: list[float] = []  # self.list_x と同一になってしまうかもしれない
        self.list_vwap: list[float] = []
        self.list_ma_1: list[float] = []
        self.list_disparity: list[float] = []

        # 銘柄コード別設定ファイルの取得
        dict_setting: dict[str, Any] = load_setting(res, code)

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
        self.dock.changedDisparityState.connect(self.switch_chart)
        self.dock.clickedSave.connect(self.on_save)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

        # ---------------------------------------------------------------------
        # チャート・インスタンス
        # ---------------------------------------------------------------------
        self.trend = trend = TrendChart(res, dict_ts, dict_setting)
        self.setCentralWidget(trend)

        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # 売買モデル用スレッド
        self.thread = QThread(self)

        # 学習済みモデルのパス
        # path_model = get_trained_ppo_model_path(res, code)

        # ワーカースレッドの生成
        self.worker = worker = WorkerAgent(code, dict_setting)
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
            self.logger.info(f"{__name__}: スレッドの終了を開始します。")

            # ワーカーにクリーンアップを実行させる
            self.requestCleanup.emit()

            # 少し待ってクリーンアップが完了するのを待つ
            QThread.msleep(100)

            # スレッドに終了を要求
            self.thread.quit()

            # タイムアウト付きで待機（5秒）
            if not self.thread.wait(5000):
                self.logger.warning(f"{__name__}: スレッドが5秒以内に応答しませんでした。強制終了します。")
                self.thread.terminate()
                self.thread.wait(1000)

            self.logger.info(f"{__name__}: スレッドを終了しました。")

        event.accept()

    def getTimePrice(self) -> pd.DataFrame:
        """
        保持している時刻、株価情報をデータフレームで返す。
        :return:
        """
        return pd.DataFrame({
            "Time": self.list_x,
            "Price": self.list_y,
            "Volume": self.list_v,
        })

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
                f"{__name__}: trade rule violation! action={action_enum}, pos={position}"
            )
            return

        # dock のメソッドを取得して実行
        getattr(self.dock, method_name)()

    def on_save(self) -> None:
        """
        チャートを保存
        :return:
        """
        if self.dock.isDisparityChecked():
            # 株価/MA1 - VWAP 乖離度のトレンドチャート
            suffix = "2"
        else:
            # 株価/MA1, VWAP トレンドチャート
            suffix = "1"
        # 　保存先のパス
        file_img = f"{self.code}_trend_{suffix}.png"
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

    def on_technicals(self, dict_technicals: dict[str, Any]) -> None:
        if dict_technicals["warmup"]:
            self.dock.trading.lockButtons()
        else:
            self.dock.trading.unLockButtons()

        # テクニカル指標
        self.vwap = dict_technicals["vwap"]
        self.list_ts.append(dict_technicals["ts"])
        self.list_ma_1.append(dict_technicals["ma1"])
        self.list_vwap.append(self.vwap)
        self.list_disparity.append(dict_technicals["ma1"] - self.vwap)

        # クロス時の縦線表示
        if 0 < dict_technicals["cross1"]:
            self.trend.setCrossGolden(dict_technicals["ts"])
        elif dict_technicals["cross1"] < 0:
            self.trend.setCrossDead(dict_technicals["ts"])

        self.update_technicals(self.dock.isDisparityChecked())

    def update_technicals(self, flag: bool) -> None:
        if flag:
            self.trend.setTechnicals(
                self.list_ts,
                [],
                [],
                self.list_disparity,
            )
        else:
            self.trend.setTechnicals(
                self.list_ts,
                self.list_ma_1,
                self.list_vwap,
                [],
            )

    def switch_chart(self, flag: bool) -> None:
        if len(self.list_x) > 0:
            ts = self.list_x[-1]
            price = self.list_y[-1]
        else:
            return

        if flag:
            self.trend.setLine([], [])
            if self.vwap > 0:
                self.trend.setDot([ts], [price - self.vwap])
            else:
                self.trend.setDot([ts], [price - self.vwap])
        else:
            self.trend.setLine(self.list_x, self.list_y)
            self.trend.setDot([ts], [price])

        # テクニカルデータの更新
        self.update_technicals(flag)

        # y 軸のスケールを更新
        self.trend.updateYAxisRange(flag)

    def on_trading_completed(self) -> None:
        self.logger.info("取引が終了しました。")

    def reset_env_completed(self) -> None:
        """
        環境をリセット済
        :return:
        """
        msg = f"{__name__}: 銘柄コード {self.code} 用の環境がリセットされました。"
        self.logger.info(msg)

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

        # リストに保持
        self.list_x.append(ts)
        self.list_y.append(price)
        self.list_v.append(volume)

        # 株価トレンド線
        flag = self.dock.isDisparityChecked()
        self.trend.setZeroLine(flag)
        if flag:
            self.trend.setLine([], [])
            if self.vwap > 0:
                self.trend.setDot([ts], [price - self.vwap])
            else:
                self.trend.setDot([ts], [price - self.vwap])
        else:
            self.trend.setLine(self.list_x, self.list_y)
            self.trend.setDot([ts], [price])

        # 銘柄単位の現在株価および含み益と収益を更新
        self.dock.setPrice(price)
        self.dock.setProfit(profit)
        self.dock.setTotal(total)

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

    def setChartTitle(self, title: str) -> None:
        """
        チャートのタイトルを設定
        :param title:
        :return:
        """
        self.trend.setTrendTitle(title)

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
