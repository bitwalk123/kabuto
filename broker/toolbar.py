import json
import os

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QToolBar

from structs.res import AppRes
from widgets.buttons import Button
from widgets.containers import Widget, PadH
from widgets.entries import EntryAddress, EntryPort
from widgets.labels import LabelRaised
from widgets.layouts import GridLayout


class ToolBarBrokerServer(QToolBar):
    def __init__(self, res: AppRes):
        super().__init__()
        self.res = res

        base = Widget()
        self.addWidget(base)

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

    def setAddressPort(self, address: str, port: int):
        self.ent_addr.setAddress(address)
        self.ent_port.setPort(port)

    def setClear(self):
        self.ent_addr.setClear()
        self.ent_port.setClear()


class ToolBarBrokerClient(QToolBar):
    requestConnectToServer = Signal(str, int)
    requestPortfolioUpdate = Signal()

    def __init__(self, res: AppRes):
        super().__init__()
        self.res = res

        with open(os.path.join(res.dir_conf, "server.json")) as f:
            dict_server = json.load(f)

        base = Widget()
        self.addWidget(base)

        layout_row = GridLayout()
        base.setLayout(layout_row)

        row = 0
        lab_server = LabelRaised("Server")
        lab_server.setFixedWidth(80)
        layout_row.addWidget(lab_server, row, 0, 2, 1)

        lab_addr = LabelRaised("Address")
        layout_row.addWidget(lab_addr, row, 1)

        lab_port = LabelRaised("Port")
        layout_row.addWidget(lab_port, row, 2)

        but_connect = Button("Connect")
        but_connect.setFixedWidth(80)
        but_connect.clicked.connect(self.connect_to_server)
        layout_row.addWidget(but_connect, row, 3, 2, 1)

        padh = PadH()
        layout_row.addWidget(padh, row, 4, 2, 1)

        self.but_update = but_update = Button("Update")
        but_update.setFixedWidth(80)
        but_update.setDisabled(True)
        but_update.clicked.connect(self.requestPortfolioUpdate.emit)
        layout_row.addWidget(but_update, row, 5, 2, 1)

        row += 1
        self.ent_addr = ent_addr = EntryAddress(dict_server["ip"])
        layout_row.addWidget(ent_addr, row, 1)

        self.ent_port = ent_port = EntryPort(str(dict_server["port"]))
        layout_row.addWidget(ent_port, row, 2)

    def connect_to_server(self):
        addr = self.ent_addr.getAddress()
        port = self.ent_port.getPort()
        self.requestConnectToServer.emit(addr, port)

    def updateEnable(self):
        self.but_update.setEnabled(True)