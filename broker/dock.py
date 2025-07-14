import logging

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QDockWidget,
    QScrollArea,
)

from structs.res import AppRes
from widgets.buttons import ButtonTicker
from widgets.containers import Widget
from widgets.layouts import VBoxLayout


class DockPortfolio(QDockWidget):
    tickerSelected = Signal(str, str)

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

        self.but_group = but_group = QButtonGroup()
        but_group.buttonToggled.connect(self.selection_changed)

    def refreshTickerList(self, list_ticker: list, dict_name: dict):
        for but in self.but_group.buttons():
            self.but_group.removeButton(but)
        for i in reversed(range(self.layout.count())):
            self.layout.itemAt(i).widget().setParent(None)
        for ticker in list_ticker:
            but = ButtonTicker(ticker, dict_name[ticker])
            self.but_group.addButton(but)
            self.layout.addWidget(but)

    def selection_changed(self, button: ButtonTicker, state):
        if state:
            ticker = button.getTicker()
            name = button.getName()
            self.tickerSelected.emit(ticker, name)
