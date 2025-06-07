import pandas as pd
import sys
from PySide6.QtWidgets import (
    QApplication,
    QHeaderView,
    QMainWindow,
    QStatusBar,
    QTableView,
)

from samples.sample_010_tablemodel import PandasModel
from widgets.labels import LabelRight, LabelPrice


class Example(QMainWindow):
    def __init__(self):
        super().__init__()
        df = pd.read_pickle('sample.pkl')

        self.setWindowTitle('取引履歴')
        self.resize(600, 400)

        view = QTableView()
        view.setStyleSheet("""
            QTableView {
                font-family: monospace;
            }
        """)
        view.setAlternatingRowColors(True)
        self.setCentralWidget(view)

        model = PandasModel(df)
        view.setModel(model)

        header = view.horizontalHeader()
        header.setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )

        statusbar = QStatusBar()
        self.setStatusBar(statusbar)

        lab_total = LabelRight("合計収益")
        statusbar.addWidget(lab_total, stretch=1)

        lab_price = LabelPrice(df["損益"].sum())
        statusbar.addWidget(lab_price)


def main():
    app = QApplication(sys.argv)
    win = Example()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
