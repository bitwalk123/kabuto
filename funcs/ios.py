import openpyxl
import pandas as pd


def load_excel(excel_path) -> dict:
    """
    excel_path で指定された Excel ファイルの読み込み
    :param excel_path:
    :return:
    """
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


def save_dataframe_to_excel(name_excel: str, dict_df: dict):
    with pd.ExcelWriter(name_excel) as writer:
        for name_sheet in dict_df.keys():
            df = dict_df[name_sheet]
            df.to_excel(writer, sheet_name=name_sheet, index=False)
