import openpyxl
import pandas as pd


def load_excel(excel_path) -> dict:
    wb = openpyxl.load_workbook(excel_path)
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
    return dict_sheet
