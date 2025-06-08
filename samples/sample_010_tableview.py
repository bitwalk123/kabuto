import pandas as pd
import sys
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow, QToolBar,
)

from structs.res import AppRes
from widgets.model import ModelTransaction
from widgets.statusbar import TotalBar
from widgets.table import TransactionView
from widgets.toolbar import ToolBarTransaction


class Example(QMainWindow):
    def __init__(self):
        super().__init__()
        res = AppRes()
        df = pd.read_pickle('sample.pkl')

        self.setWindowTitle('取引履歴')
        self.resize(600, 400)

        toolbar = ToolBarTransaction(res)
        self.addToolBar(toolbar)

        view = TransactionView()
        self.setCentralWidget(view)

        model = ModelTransaction(df)
        view.setModel(model)

        statusbar = TotalBar()
        self.setStatusBar(statusbar)

        total = df["損益"].sum()
        statusbar.setTotal(total)


def main():
    app = QApplication(sys.argv)
    win = Example()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
