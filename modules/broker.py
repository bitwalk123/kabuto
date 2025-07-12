import json
import logging
import os

from PySide6.QtGui import QIcon
from PySide6.QtNetwork import (
    QHostAddress,
    QTcpServer,
    QTcpSocket,
)
from PySide6.QtWidgets import (
    QMainWindow,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from structs.res import AppRes


class StockBroker(QMainWindow):
    def __init__(self):
        super().__init__()
        # モジュール固有のロガーを取得
        self.logger = logging.getLogger(__name__)
        self.res = res = AppRes()

        # json でサーバー情報を取得（ポート番号のみ使用）
        with open(os.path.join(res.dir_conf, "server.json")) as f:
            dict_server = json.load(f)

        self.client: QTcpSocket | None = None
        self.server = QTcpServer(self)
        self.server.listen(QHostAddress.SpecialAddress.Any, dict_server["port"])
        self.server.newConnection.connect(self.new_connection)

        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # UI
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        self.resize(400, 300)
        icon = QIcon(os.path.join(res.dir_image, "bee.png"))
        self.setWindowIcon(icon)
        self.setWindowTitle("StockBroker")

        base = QWidget()
        self.setCentralWidget(base)

        layout = QVBoxLayout()
        base.setLayout(layout)

        self.tedit = tedit = QTextEdit(self)
        tedit.setStyleSheet("QTextEdit {font-family: monospace;}")
        tedit.setReadOnly(True)  # Set it to read-only for history
        layout.addWidget(tedit)

    def disconnected_connection(self, *args):
        print("DEBUG!", *args)

    def new_connection(self):
        self.client = self.server.nextPendingConnection()
        self.client.readyRead.connect(self.receive_message)
        self.client.disconnected.connect(self.disconnected_connection)

        # ピア情報
        peerAddress = self.client.peerAddress()
        peerPort = self.client.peerPort()
        peerInfo = f"{peerAddress.toString()}:{peerPort}"
        self.logger.info(f"{__name__} Connected from {peerInfo}.")
        self.client.write(f"Server connected from {peerInfo}".encode())

    def receive_message(self):
        msg = self.client.readAll().data().decode()
        self.tedit.append(f"Received: {msg}")
        # just for verification
        self.client.write(f"Server received: {msg}".encode())
