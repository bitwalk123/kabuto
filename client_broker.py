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

        self.socket = QTcpSocket(self)
        self.socket.connected.connect(self.connecting)
        self.socket.disconnected.connect(self.connection_lost)
        self.socket.readyRead.connect(self.receive_message)

        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # UI
        self.setWindowTitle("Client")

        toolbar = ToolBarBrokerClient(res)
        toolbar.requestConnectToServer.connect(self.connect_to_server)
        self.addToolBar(toolbar)

        base = Widget()
        self.setCentralWidget(base)

        layout = VBoxLayout()
        base.setLayout(layout)

        self.log_win = log_win = MultilineLog()
        layout.addWidget(log_win)

        statusbar = StatusBarBrokerClient(res)
        statusbar.requestSendMessage.connect(self.send_message)
        self.setStatusBar(statusbar)
        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_

    def connect_to_server(self, addr: str, port: int):
        self.socket.connectToHost(addr, port)

    def connection_lost(self):
        self.log_win.append("Server disconnected.")

    def connecting(self):
        self.log_win.append("Connecting to server...")

    def receive_message(self):
        s = self.socket.readAll().data().decode()
        dict_msg = json.loads(s)
        if "message" in dict_msg.keys():
            self.log_win.append(f'Received: {dict_msg["message"]}')

    def send_message(self, msg: str):
        if msg:
            self.log_win.append(f"Sent: {msg}")
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
