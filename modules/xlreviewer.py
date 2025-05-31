import logging

from PySide6.QtCore import QObject, Signal

from funcs.io import load_excel


class ExcelReviewer(QObject):
    # スレッド完了シグナル（成否の論理値）
    threadFinished = Signal(bool)

    def __init__(self, excel_path: str):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.excel_path = excel_path

    def run(self):
        try:
            dict_sheet = load_excel(self.excel_path)
        except Exception as e:
            self.logger.critical(f"Excelファイルの読み込み中にエラーが発生しました: {e}")
            self.threadFinished.emit(False)
            return

        print(dict_sheet)

        # 　スレッドの終了
        self.threadFinished.emit(True)
