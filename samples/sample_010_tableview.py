import pandas as pd
import sys
from PySide6.QtWidgets import (
    QApplication,
    QHeaderView,
    QMainWindow,
    QTableView,
)

from samples.sample_010_tablemodel import PandasModel


class Example(QMainWindow):
    def __init__(self):
        super().__init__()
        df = pd.read_pickle('sample.pkl')
        self.setWindowTitle('QTableView')
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


def main():
    app = QApplication(sys.argv)
    win = Example()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
