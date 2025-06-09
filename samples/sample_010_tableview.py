import math
import os

import numpy as np
import pandas as pd
import sys
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow, QFileDialog,
)

from structs.res import AppRes
from widgets.model import ModelTransaction
from widgets.statusbar import TotalBar
from widgets.table import TransactionView
from widgets.toolbar import ToolBarTransaction


class Example(QMainWindow):
    def __init__(self):
        super().__init__()
        self.res = res = AppRes()
        self.df = df = pd.read_pickle('sample.pkl')

        self.setWindowTitle('取引履歴')
        self.resize(600, 400)

        toolbar = ToolBarTransaction(res)
        toolbar.saveClicked.connect(self.on_save_dlg)
        toolbar.excelSelected.connect(self.on_excel_selected)
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

    def on_excel_selected(self, path_excel: str):
        df = pd.read_excel(path_excel)

        print("<table>")
        print("<thead>")
        print("<tr>")
        for colname in df.columns:
            print(f"<th>{colname}</th>")
        print("</tr>")
        print("</thead>")
        print("<tbody")
        for rowname in df.index:
            print("<tr>")
            for colname in df.columns:
                cell = df.at[rowname, colname]
                if pd.isna(cell):
                    cell = ""
                match colname:
                    case "注文番号":
                        print(f'<td style="text-align: right;">{cell}</td>')
                    case "注文日時":
                        print(f'<td style="text-align: center;">{cell}</td>')
                    case "銘柄コード":
                        print(f'<td style="text-align: center;">{cell}</td>')
                    case "売買":
                        print(f'<td style="text-align: center;">{cell}</td>')
                    case "約定単価":
                        print(f'<td style="text-align: right;">{cell}</td>')
                    case "約定数量":
                        print(f'<td style="text-align: right;">{cell}</td>')
                    case "損益":
                        print(f'<td style="text-align: right;">{cell}</td>')
                    case "備考":
                        print(f'<td style="text-align: left;">{cell}</td>')
            print("</tr>")

        total = df["損益"].sum()
        print("<tr>")
        # 注文番号
        print('<td style="text-align: right;"></td>')
        # 注文日時
        print('<td style="text-align: center;"></td>')
        # 銘柄コード
        print('<td style="text-align: center;"></td>')
        # 売買
        print('<td style="text-align: center;"></td>')
        # 約定単価, 約定数量
        print('<td style="text-align: right;" colspan="2">合計損益</td>')
        # 損益
        print(f'<td style="text-align: right;">{total}</td>')
        # 備考"
        print('<td style="text-align: left;"></td>')
        print("</tr>")
        print("</tbody>")
        print("</table>")


def main():
    app = QApplication(sys.argv)
    win = Example()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
