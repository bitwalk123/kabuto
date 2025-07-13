# Reference:
# https://github.com/bhowiebkr/client-server-socket-example/
import sys

from PySide6.QtWidgets import (
    QApplication,
    QFormLayout,
    QLineEdit,
    QMainWindow,
    QTextEdit,
)
from PySide6.QtNetwork import QTcpSocket

from broker.toolbar import ToolBarBrokerClient
from structs.res import AppRes
from widgets.containers import Widget
from widgets.layouts import VBoxLayout


class TcpSocketClient(QMainWindow):
    def __init__(self):
        super().__init__()
        self.res = res = AppRes()

        self.socket = QTcpSocket(self)
        self.socket.connected.connect(self.connecting)
        self.socket.disconnected.connect(self.connection_lost)
        self.socket.readyRead.connect(self.receive_message)

        # UI
        self.setWindowTitle("Client")

        toolbar = ToolBarBrokerClient(res)
        toolbar.requestConnectToServer.connect(self.connect_to_server)
        self.addToolBar(toolbar)

        base = Widget()
        self.setCentralWidget(base)

        layout = VBoxLayout()
        base.setLayout(layout)

        self.tedit = tedit = QTextEdit(self)
        tedit.setStyleSheet("QTextEdit {font-family: monospace;}")
        tedit.setReadOnly(True)  # Set it to read-only for history
        layout.addWidget(tedit)

        self.ledit = ledit = QLineEdit(self)
        ledit.returnPressed.connect(self.send_message)  # Send when Return key is pressed
        form = QFormLayout()
        form.addRow("Message:", ledit)
        layout.addLayout(form)

    def connect_to_server(self, addr: str, port: int):
        self.socket.connectToHost(addr, port)

    def connection_lost(self):
        self.tedit.append("Disconnected.")

    def connecting(self):
        self.tedit.append("Connecting to server...")

    def receive_message(self):
        msg = self.socket.readAll().data().decode()
        self.tedit.append(f"Received: {msg}")

    def send_message(self):
        msg = self.ledit.text()
        if msg:
            self.tedit.append(f"Sent: {msg}")
            self.socket.write(msg.encode())
            self.ledit.clear()  # Clear the input field after sending


def main():
    app = QApplication(sys.argv)
    win = TcpSocketClient()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
