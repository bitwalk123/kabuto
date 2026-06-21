import os
import re

import pandas as pd
from PySide6.QtCore import Signal
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QToolBar

from funcs.tse import get_ticker_name_list
from structs.res import AppRes
from widgets.containers import PadH
from widgets.dialogs import DlgCSVFileSel


class ProfitSimulatorToolbar(QToolBar):
    sendDataFrame = Signal(pd.DataFrame, str, str)
    pattern_code = re.compile(r".*([0-9A-X]{4})_.+\.csv")

    def __init__(self, res: AppRes):
        super().__init__()
        self.res = res
        self.path_csv: str = ""
        self.code = "0000"

        # 出力された CSV ファイルを開く
        self.csv = action_open = QAction(
            QIcon(os.path.join(res.dir_image, "csv.png")),
            "CSV ファイルを開く",
            self
        )
        action_open.triggered.connect(self.on_select_technicals)
        self.addAction(action_open)

        pad = PadH()
        self.addWidget(pad)

    def getCode(self) -> str:
        """
        保持している銘柄コードの取得
        :return:
        """
        return self.code

    def on_select_technicals(self):
        """
        ティックデータを保持した Excel ファイルの選択
        :return:
        """
        # ティックデータ（Excel ファイル）の選択ダイアログ
        dlg_file = DlgCSVFileSel(self.res)
        if dlg_file.exec():
            self.path_csv = dlg_file.selectedFiles()[0]
        else:
            return

        df = pd.read_csv(self.path_csv, index_col=0, parse_dates=True)

        # チャートのタイトル文字列
        d_str = df.index[0].strftime('%Y-%m-%d')
        if m := self.pattern_code.match(self.path_csv):
            self.code = m.group(1)
        name = get_ticker_name_list([self.code])[self.code]
        title = f"{d_str} : {name} ({self.code})"

        self.sendDataFrame.emit(df, title, self.path_csv)
