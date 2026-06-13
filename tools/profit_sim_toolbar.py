import os

import pandas as pd
from PySide6.QtCore import Signal
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QToolBar

from structs.res import AppRes
from widgets.dialogs import DlgOutputFileSel


class ProfitSimulatorToolbar(QToolBar):
    sendDataFrame = Signal(pd.DataFrame)

    def __init__(self, res: AppRes):
        super().__init__()
        self.res = res

        # 出力された CSV ファイルを開く
        self.csv = action_open = QAction(
            QIcon(os.path.join(res.dir_image, "csv.png")),
            "CSV ファイルを開く",
            self
        )
        action_open.triggered.connect(self.on_select_output)
        self.addAction(action_open)

    def on_select_output(self):
        """
        ティックデータを保持した Excel ファイルの選択
        :return:
        """
        # ティックデータ（Excel ファイル）の選択ダイアログ
        dlg_file = DlgOutputFileSel(self.res)
        if dlg_file.exec():
            path_csv = dlg_file.selectedFiles()[0]
        else:
            return

        df = pd.read_csv(path_csv, index_col=0, parse_dates=True)
        # rint(df)
        # print(df.index.dtype)
        self.sendDataFrame.emit(df)
