"""
取引履歴表示

機能スコープ
1. 取引履歴をテーブルに表示
2. Excel / HTML 形式で保存
"""
import pandas as pd
from PySide6.QtWidgets import QMainWindow

from structs.res import AppRes
from widgets.model import ModelTransaction
from widgets.statusbar import TotalBar
from widgets.table import TransactionView


class WinTransaction(QMainWindow):
    def __init__(self, res: AppRes, df: pd.DataFrame):
        super().__init__()
        self.res = res

        self.resize(600, 600)
        self.setWindowTitle("取引履歴")

        view = TransactionView()
        self.setCentralWidget(view)

        model = ModelTransaction(df)
        view.setModel(model)

        statusbar = TotalBar()
        self.setStatusBar(statusbar)

        total = df["損益"].sum()
        statusbar.setTotal(total)
