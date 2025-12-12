import json
import os

import openpyxl
import pandas as pd


def is_sheet_exists(path_excel: str, sheet: str) -> bool:
    """
    指定したExcelファイルに指定したシートが存在するかどうか確認
    :param path_excel:
    :param sheet:
    :return:
    """
    if os.path.isfile(path_excel):
        wb = pd.ExcelFile(path_excel)
        list_sheet = wb.sheet_names
        if sheet in list_sheet:
            return True
        else:
            return False
    else:
        return False


def get_excel_sheet(path_excel: str, sheet: str) -> pd.DataFrame:
    """
    指定したExcelファイルの指定したシートをデータフレームに読み込む
    :param path_excel:
    :param sheet:
    :return:
    """
    if os.path.isfile(path_excel):
        wb = pd.ExcelFile(path_excel)
        list_sheet = wb.sheet_names
        if sheet in list_sheet:
            return wb.parse(sheet_name=sheet)
        else:
            return pd.DataFrame()
    else:
        return pd.DataFrame()


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


def read_contents_from_json(file_json) -> dict:
    with open(file_json) as f:
        dict_psar = json.load(f)
    return dict_psar


def save_contents_to_json(file_json: str, dict_psar: dict):
    with open(file_json, "w") as f:
        json.dump(dict_psar, f)
