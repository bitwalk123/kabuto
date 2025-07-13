# Reference:
# https://github.com/bhowiebkr/client-server-socket-example/
import json
import sys

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
)
from PySide6.QtNetwork import QTcpSocket

from broker.statusbar import StatusBarBrokerClient
from broker.toolbar import ToolBarBrokerClient
from structs.res import AppRes
from widgets.containers import Widget
from widgets.layouts import VBoxLayout
from widgets.textedit import MultilineLog


class TcpSocketClient(QMainWindow):
    def __init__(self):
        super().__init__()
        self.res = res = AppRes()

        self.list_ticker = list()
        self.dict_name = dict()

        self.socket = QTcpSocket(self)
        self.socket.connected.connect(self.connecting)
        self.socket.disconnected.connect(self.connection_lost)
        self.socket.readyRead.connect(self.receive_message)

        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # UI
        self.setWindowTitle("Client")

        self.toolbar = toolbar = ToolBarBrokerClient(res)
        toolbar.requestConnectToServer.connect(self.connect_to_server)
        toolbar.requestPortfolioUpdate.connect(self.request_portfolio)
        self.addToolBar(toolbar)

        # base = Widget()
        # self.setCentralWidget(base)

        # layout = VBoxLayout()
        # base.setLayout(layout)

        # self.log_win = log_win = MultilineLog()
        # layout.addWidget(log_win)

        statusbar = StatusBarBrokerClient(res)
        statusbar.requestSendMessage.connect(self.send_message)
        self.setStatusBar(statusbar)
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_

    def connect_to_server(self, addr: str, port: int):
        self.socket.connectToHost(addr, port)

    def connection_lost(self):
        print("Server disconnected.")

    def connecting(self):
        print("Connecting to server...")

    def receive_message(self):
        s = self.socket.readAll().data().decode()
        d = json.loads(s)

        if "message" in d.keys():
            print(f'Received: {d["message"]}')

        if "connection" in d.keys():
            print(d["connection"])
            self.toolbar.updateEnable()

        if "portfolio" in d.keys():
            print("Received updated portfolio.")
            if "list_ticker" in d["portfolio"].keys():
                self.list_ticker = sorted(d["portfolio"]["list_ticker"])
                if "dict_name" in d["portfolio"].keys():
                    self.dict_name = d["portfolio"]["dict_name"]
            print(self.list_ticker)
            print(self.dict_name)

    def request_portfolio(self):
        dict_request = {"request": "portfolio"}
        s = json.dumps(dict_request)
        self.socket.write(s.encode())

    def send_message(self, msg: str):
        if msg:
            print(f"Sent: {msg}")
            dict_msg = {"message": msg}
            s = json.dumps(dict_msg)
            self.socket.write(s.encode())


def main():
    app = QApplication(sys.argv)
    win = TcpSocketClient()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
