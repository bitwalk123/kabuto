"""
取引履歴表示

機能スコープ
1. 取引履歴をテーブルに表示
2. Excel 形式で保存
3. HTML 形式で出力
"""
import os

import pandas as pd
from PySide6.QtWidgets import QMainWindow, QFileDialog

from funcs.conv import conv_transaction_df2html
from structs.res import AppRes
from widgets.models import ModelTransaction
from widgets.statusbars import TotalBar
from widgets.tables import TransactionView
from widgets.toolbars import ToolBarTransaction


class WinTransaction(QMainWindow):
    def __init__(self, res: AppRes, df: pd.DataFrame):
        super().__init__()
        self.res = res
        self.df = df

        self.resize(600, 600)
        self.setWindowTitle("取引履歴")

        toolbar = ToolBarTransaction(res)
        toolbar.saveClicked.connect(self.on_save_dlg)
        toolbar.transdataSelected.connect(self.on_excel_transaction_selected)
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

    def on_excel_transaction_selected(self, path_excel: str):
        df = pd.read_excel(path_excel)
        list_html = conv_transaction_df2html(df)
        for line in list_html:
            print(line)
