import logging

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDockWidget, QWidget

from structs.res import AppRes
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

        base = Widget()
        self.setWidget(base)

        self.layout = layout = VBoxLayout()
        layout.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        layout.setSpacing(2)
        base.setLayout(layout)

    def refreshTickerList(self, list_ticker: list, dict_name: dict):
        for ticker in list_ticker:
            print(f"{dict_name[ticker]} ({ticker})")
