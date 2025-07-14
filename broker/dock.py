import logging

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDockWidget, QWidget, QScrollArea

from structs.res import AppRes
from widgets.buttons import ButtonTicker
from widgets.containers import Widget
from widgets.layouts import VBoxLayout


class DockPortfolio(QDockWidget):
    def __init__(self, res: AppRes):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.res = res

        self.setFeatures(
            QDockWidget.DockWidgetFeature.NoDockWidgetFeatures
        )
        self.setTitleBarWidget(Widget())

        sa = QScrollArea()
        sa.setWidgetResizable(True)
        self.setWidget(sa)

        base = Widget()
        sa.setWidget(base)

        self.layout = layout = VBoxLayout()
        layout.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight
        )
        layout.setSpacing(0)
        base.setLayout(layout)

    def refreshTickerList(self, list_ticker: list, dict_name: dict):
        for i in reversed(range(self.layout.count())):
            self.layout.itemAt(i).widget().setParent(None)
        for ticker in list_ticker:
            but = ButtonTicker(ticker, dict_name[ticker])
            self.layout.addWidget(but)
