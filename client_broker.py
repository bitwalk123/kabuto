# Reference:
# https://github.com/bhowiebkr/client-server-socket-example/
import json
import os
import sys

from PySide6.QtWidgets import (
    QApplication,
    QFormLayout,
    QLineEdit,
    QMainWindow,
    QTextEdit,
)
from PySide6.QtNetwork import QTcpSocket

from structs.res import AppRes
from widgets.buttons import Button
from widgets.containers import Widget
from widgets.entries import EntryAddress, EntryPort
from widgets.labels import LabelRaised
from widgets.layouts import GridLayout, VBoxLayout


class TcpSocketClient(QMainWindow):
    def __init__(self):
        super().__init__()
        self.res = res = AppRes()
        with open(os.path.join(res.dir_conf, "server.json")) as f:
            dict_server = json.load(f)

        self.socket = QTcpSocket(self)
        self.socket.connected.connect(self.connecting)
        self.socket.disconnected.connect(self.connection_lost)
        self.socket.readyRead.connect(self.receive_message)

        # UI
        self.setWindowTitle("Client")

        base = Widget()
        self.setCentralWidget(base)

        layout = VBoxLayout()
        base.setLayout(layout)

        layout_row = GridLayout()
        layout.addLayout(layout_row)

        row = 0
        lab_server = LabelRaised("Server")
        lab_server.setFixedWidth(60)
        layout_row.addWidget(lab_server, row, 0, 2, 1)

        lab_addr = LabelRaised("Address")
        layout_row.addWidget(lab_addr, row, 1)

        lab_port = LabelRaised("Port")
        layout_row.addWidget(lab_port, row, 2)

        but_connect = Button("Connect")
        but_connect.clicked.connect(self.connect_to_server)
        layout_row.addWidget(but_connect, row, 3, 2, 1)

        row += 1
        self.ent_addr = ent_addr = EntryAddress(dict_server["ip"])
        layout_row.addWidget(ent_addr, row, 1)

        self.ent_port = ent_port = EntryPort(str(dict_server["port"]))
        layout_row.addWidget(ent_port, row, 2)

        self.tedit = tedit = QTextEdit(self)
        tedit.setStyleSheet("QTextEdit {font-family: monospace;}")
        tedit.setReadOnly(True)  # Set it to read-only for history
        layout.addWidget(tedit)

        self.ledit = ledit = QLineEdit(self)
        ledit.returnPressed.connect(self.send_message)  # Send when Return key is pressed
        form = QFormLayout()
        form.addRow("Message:", ledit)
        layout.addLayout(form)

    def connect_to_server(self):
        self.socket.connectToHost(
            self.ent_addr.text(),
            int(self.ent_port.text())
        )

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
