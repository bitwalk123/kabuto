import logging

import openpyxl
import pandas as pd
from PySide6.QtCore import QObject, Signal

class ExcelLoader(QObject):
    # スレッド完了シグナル（成否の論理値）
    threadFinished = Signal(bool)
    # エラーシグナル
    errorOccurred = Signal()

    def __init__(self, excel_path: str):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.excel_path = excel_path

    def run(self):
        try:
            wb = openpyxl.load_workbook(self.excel_path)
            dict_sheet = dict()

            for i, name_sheet in enumerate(wb.sheetnames):
                sheet = wb[name_sheet]
                data = sheet.values
                # 最初の行をヘッダーとしてPandasのDataFrameに読み込む
                # openpyxlから直接DataFrameを作成すると、空のセルがNoneになるため、適宜処理が必要な場合がある
                columns = next(data, None)  # ヘッダー行がない場合の対応
                if columns:
                    df = pd.DataFrame(data, columns=columns)
                else:  # ヘッダー行がない場合は、データ部分からDataFrameを作成
                    df = pd.DataFrame(data)
                dict_sheet[name_sheet] = df
            wb.close()

        except Exception as e:
            self.logger.critical(f"Excelファイルの読み込み中にエラーが発生しました: {e}")
            self.errorOccurred.emit()
            self.threadFinished.emit(False)

        #　スレッドの終了
        self.threadFinished.emit(True)
