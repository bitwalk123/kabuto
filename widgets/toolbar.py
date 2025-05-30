from PySide6.QtCore import Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QStyle, QToolBar, QFileDialog

from structs.res import AppRes


class ToolBar(QToolBar):
    excelSelected = Signal(str)

    def __init__(self, res: AppRes):
        super().__init__()
        self.res = res

        if res.debug:
            action_open = QAction(
                self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon),
                'Excel ファイルを開く',
                self
            )
            action_open.triggered.connect(self.on_select_excel)
            self.addAction(action_open)

        action_save = QAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton),
            'データを保存する',
            self
        )
        self.addAction(action_save)

        action_info = QAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxInformation),
            'アプリケーションの情報',
            self
        )
        self.addAction(action_info)

        action_close = QAction(
            self.style().standardIcon(QStyle.StandardPixmap.SP_TabCloseButton),
            'アプリケーションの終了',
            self
        )
        self.addAction(action_close)

    def on_select_excel(self):
        excel_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open File",
            self.res.dir_excel,
            "Excel File (*.xlsx)"
        )
        if excel_path == "":
            return
        else:
            self.excelSelected.emit(excel_path)