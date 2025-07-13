import json
import logging
import os

from PySide6.QtGui import QIcon
from PySide6.QtNetwork import (
    QHostAddress,
    QTcpServer,
    QTcpSocket,
)
from PySide6.QtWidgets import QMainWindow

from structs.res import AppRes
from widgets.containers import Widget
from widgets.entries import EntryAddress, EntryPort
from widgets.labels import LabelRaised
from widgets.layouts import GridLayout


class StockBroker(QMainWindow):
    def __init__(self):
        super().__init__()
        # モジュール固有のロガーを取得
        self.logger = logging.getLogger(__name__)
        self.res = res = AppRes()
        # json でサーバー情報を取得（ポート番号のみ使用）
        with open(os.path.join(res.dir_conf, "server.json")) as f:
            dict_server = json.load(f)
        # サーバー・インスタンス
        self.server = QTcpServer(self)
        self.server.listen(QHostAddress.SpecialAddress.Any, dict_server["port"])
        self.server.newConnection.connect(self.connection_new)
        # クライアント・インスタンス（常に１つのみで運用）
        self.client: QTcpSocket | None = None

        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        #  UI
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        icon = QIcon(os.path.join(res.dir_image, "bee.png"))
        self.setWindowIcon(icon)
        self.setWindowTitle("StockBroker")

        base = Widget()
        self.setCentralWidget(base)

        layout = GridLayout()
        base.setLayout(layout)

        row = 0
        lab_client = LabelRaised("Client")
        lab_client.setFixedWidth(60)
        layout.addWidget(lab_client, row, 0, 2, 1)

        lab_addr = LabelRaised("Address")
        layout.addWidget(lab_addr, row, 1)

        lab_port = LabelRaised("Port")
        layout.addWidget(lab_port, row, 2)

        row += 1
        self.ent_addr = ent_addr = EntryAddress()
        layout.addWidget(ent_addr, row, 1)

        self.ent_port = ent_port = EntryPort()
        layout.addWidget(ent_port, row, 2)

    def connection_lost(self):
        self.logger.info(f"{__name__} Client disconnected.")
        # クライアントの切断処理
        self.client = None
        self.ent_addr.setClear()
        self.ent_port.setClear()
        # 接続待ちがあれば新しい接続処理へ
        if self.server.hasPendingConnections():
            self.connection_new()

    def connection_new(self):
        if self.client is None:
            self.client = self.server.nextPendingConnection()
            self.client.readyRead.connect(self.receive_message)
            self.client.disconnected.connect(self.connection_lost)
            # ピア情報
            peerAddress = self.client.peerAddress().toString()
            peerPort = self.client.peerPort()
            self.ent_addr.setAddress(peerAddress)
            self.ent_port.setPort(peerPort)
            # ログ出力＆クライアントへ応答
            peerInfo = f"{peerAddress}:{peerPort}"
            self.logger.info(f"{__name__} Connected from {peerInfo}.")
            self.client.write(f"Server accepted connecting from {peerInfo}".encode())
        else:
            # 一度に接続できるのは１クライアントのみに制限
            self.server.pauseAccepting()  # 接続を保留
            self.logger.warning(f"{__name__} Pause accepting new connection.")

    def receive_message(self):
        msg = self.client.readAll().data().decode()
        print(f"Received: {msg}")
        # サーバーの応答をクライアントへ
        self.client.write(f"Server received: {msg}".encode())
