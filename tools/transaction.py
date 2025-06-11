import os

import pandas as pd
import sys
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
)

from funcs.conv import conv_transaction_df2html
from structs.res import AppRes
from widgets.model import ModelTransaction
from widgets.statusbar import TotalBar
from widgets.table import TransactionView
from widgets.toolbar import ToolBarTransaction


class Transaction(QMainWindow):
    def __init__(self):
        super().__init__()
        self.res = res = AppRes()
        self.df = df = pd.read_pickle('sample.pkl')

        self.setWindowTitle('取引履歴')
        self.resize(600, 400)

        toolbar = ToolBarTransaction(res)
        toolbar.saveClicked.connect(self.on_save_dlg)
        toolbar.transdataSelected.connect(self.on_excel_tickdata_selected)
        self.addToolBar(toolbar)

        view = TransactionView()
        self.setCentralWidget(view)

        model = ModelTransaction(df)
        view.setModel(model)

        statusbar = TotalBar()
        self.setStatusBar(statusbar)

        total = df["損益"].sum()
        statusbar.setTotal(total)

    def on_save_dlg(self):
        excel_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save File",
            os.path.join(self.res.dir_transaction, "untitled.xlsx"),
            "Excel File (*.xlsx)"
        )
        if excel_path == "":
            return
        else:
            print(excel_path)
            self.df.to_excel(excel_path, index=False, header=True)

    def on_excel_tickdata_selected(self, path_excel: str):
        df = pd.read_excel(path_excel)
        list_html = conv_transaction_df2html(df)
        for line in list_html:
            print(line)


def main():
    app = QApplication(sys.argv)
    win = Transaction()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
