import pandas as pd
import sys
from PySide6.QtWidgets import (
    QApplication,
    QHeaderView,
    QMainWindow,
    QStatusBar,
    QTableView,
)

from widgets.model import ModelTransaction
from widgets.labels import LabelRight, LabelPrice
from widgets.statusbar import TotalBar
from widgets.table import TransactionView


class Example(QMainWindow):
    def __init__(self):
        super().__init__()
        df = pd.read_pickle('sample.pkl')

        self.setWindowTitle('取引履歴')
        self.resize(600, 400)

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
