import json
import logging
import os

from PySide6.QtGui import QIcon, QCloseEvent
from PySide6.QtNetwork import (
    QHostAddress,
    QTcpServer,
    QTcpSocket,
)
from PySide6.QtWidgets import QMainWindow

from broker.portfolio import Portfolio
from broker.toolbar import ToolBarBrokerServer
from structs.res import AppRes


class StockBroker(QMainWindow):
    def __init__(self):
        super().__init__()
        # モジュール固有のロガーを取得
        self.logger = logging.getLogger(__name__)
        self.res = res = AppRes()
        # ---------------------------------------------------------------------
        # json でサーバー情報を取得（ポート番号のみ使用）
        with open(os.path.join(res.dir_conf, "server.json")) as f:
            dict_server = json.load(f)
        # ---------------------------------------------------------------------
        # サーバー・インスタンス
        self.server = QTcpServer(self)
        self.server.listen(QHostAddress.SpecialAddress.Any, dict_server["port"])
        self.server.newConnection.connect(self.connection_new)
        # クライアント・インスタンス（常に１つのみで運用）
        self.client: QTcpSocket | None = None

        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        #  UI
        icon = QIcon(os.path.join(res.dir_image, "bee.png"))
        self.setWindowIcon(icon)
        self.setWindowTitle("StockBroker")
        # ツールバー
        self.toolbar = toolbar = ToolBarBrokerServer(res)
        self.addToolBar(toolbar)
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_

        # ---------------------------------------------------------------------
        # portfolio スレッド用インスタンス
        self.portfolio = portfolio = Portfolio(res)
        portfolio.threadReady.connect(self.on_portfolio_ready)
        portfolio.worker.notifyInitCompleted.connect(self.on_portfolio_init_completed)
        portfolio.worker.notifyCurrentPortfolio.connect(self.on_portfolio_current)
        portfolio.start()

    def closeEvent(self, event: QCloseEvent):
        # ---------------------------------------------------------------------
        # Thread Stock Collector の削除
        # ---------------------------------------------------------------------
        if self.portfolio.isRunning():
            self.portfolio.requestStopProcess.emit()
            self.logger.info("Stopping Portfolio...")
            self.portfolio.quit()  # スレッドのイベントループに終了を指示
            self.portfolio.wait()  # スレッドが完全に終了するまで待機
            self.logger.info("Portfolio safely terminated.")

        # ---------------------------------------------------------------------
        self.logger.info(f"{__name__} stopped and closed.")
        event.accept()

    def connection_lost(self):
        self.logger.info(f"{__name__}: Client disconnected.")
        # ---------------------------------------------------------------------
        # クライアントの切断処理
        self.client = None
        self.toolbar.setClear()
        # ---------------------------------------------------------------------
        # 接続待ちがあれば新しい接続処理へ
        if self.server.hasPendingConnections():
            self.connection_new()

    def connection_new(self):
        if self.client is None:
            # ---------------------------------------------------------------------
            # 接続処理
            self.client = self.server.nextPendingConnection()
            self.client.readyRead.connect(self.receive_json)
            self.client.disconnected.connect(self.connection_lost)
            # ---------------------------------------------------------------------
            # ピア情報
            peerAddress = self.client.peerAddress().toString()
            peerPort = self.client.peerPort()
            self.toolbar.setAddressPort(peerAddress, peerPort)
            # ---------------------------------------------------------------------
            # ログ出力＆クライアントへ応答
            peerInfo = f"{peerAddress}:{peerPort}"
            self.logger.info(f"{__name__}: Connected from {peerInfo}.")
            msg = f"Server accepted connecting from {peerInfo}"
            d = {"connection": msg}
            s = json.dumps(d)
            self.client.write(s.encode())
        else:
            # ---------------------------------------------------------------------
            # 一度に接続できるのは１クライアントのみに制限
            self.server.pauseAccepting()  # 接続を保留
            self.logger.warning(f"{__name__}: Pause accepting new connection.")

    def on_portfolio_current(self, list_code: list, dict_name: dict):
        print("Updated portfolio obtained.")
        d = {
            "portfolio": {
                "list_code": list_code,
                "dict_name": dict_name,
            }
        }
        s = json.dumps(d)
        self.client.write(s.encode())

    @staticmethod
    def on_portfolio_init_completed(list_code: list, dict_name: dict):
        """
        スレッド初期化後の銘柄リスト
        :param list_code:
        :param dict_name:
        :return:
        """
        print("### 起動時のポートフォリオ（現物） ###")
        for code in list_code:
            print(code, dict_name[code])

    def on_portfolio_ready(self):
        self.logger.info(f"{__name__}: Portfolio thread is ready.")

    def receive_json(self):
        s = self.client.readAll().data().decode()
        d = json.loads(s)
        if "message" in d.keys():
            print(f'Received: {d["message"]}')

        if "request" in d.keys():
            if d["request"] == "portfolio":
                # --------------------------------------------
                # 🧿 現在のポートフォリオの情報をリクエスト
                self.portfolio.requestCurrentPortfolio.emit()
                # --------------------------------------------

        # ---------------------------------------------------------------------
        # サーバーの応答をクライアントへ
        # self.client.write(f"Server received: {msg}".encode())
