import datetime
import json

import mplfinance as mpf
import yfinance as yf
from PySide6.QtCore import Qt
from PySide6.QtNetwork import QTcpSocket
from PySide6.QtWidgets import QMainWindow, QStatusBar

from broker.dock import DockPortfolio
from broker.statusbar import StatusBarBrokerClient
from broker.technical import Technical
from broker.toolbar import ToolBarBrokerClient
from modules.psar_conventional import ParabolicSAR
from structs.res import AppRes
from widgets.chart import CandleChart, ChartNavigation


class PortfolioViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.res = res = AppRes()

        self.socket = QTcpSocket(self)
        self.socket.connected.connect(self.connecting)
        self.socket.disconnected.connect(self.connection_lost)
        self.socket.readyRead.connect(self.receive_message)

        # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
        # UI
        self.setWindowTitle("Portfolio Viewer")
        self.resize(1500, 800)

        # ツールバー
        self.toolbar = toolbar = ToolBarBrokerClient(res)
        toolbar.requestConnectToServer.connect(self.connect_to_server)
        toolbar.requestPortfolioUpdate.connect(self.request_portfolio)
        self.addToolBar(toolbar)

        # 右側のドック
        self.dock = dock = DockPortfolio(res)
        dock.tickerSelected.connect(self.ticker_selected)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

        self.chart = chart = CandleChart()
        chart.initChart(2)
        self.setCentralWidget(chart)

        # ステータスバー
        # statusbar = StatusBarBrokerClient(res)
        # statusbar.requestSendMessage.connect(self.send_message)
        navbar = ChartNavigation(chart)
        statusbar = QStatusBar()
        statusbar.addWidget(navbar)
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
            list_code = list()
            dict_name = dict()
            if "list_ticker" in d["portfolio"].keys():
                list_code = sorted(d["portfolio"]["list_ticker"])
                if "dict_name" in d["portfolio"].keys():
                    dict_name = d["portfolio"]["dict_name"]
            # ドックに最新情報をインプット
            self.dock.refreshTickerList(list_code, dict_name)

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

    def ticker_selected(self, code: str, name: str):
        tech = Technical(self.chart)
        tech.draw(code, name)