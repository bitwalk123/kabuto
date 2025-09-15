import re

import numpy as np
import pandas as pd


def conv_date_str(date_str: str) -> str:
    """
    日付文字列 YYYYMMDD を YYYY-MM-DD に変換
    :param date_str: 日付文字列 YYYYMMDD
    :return: 日付文字列 YYYY-MM-DD
    """
    pattern = re.compile(r'(\d{4})(\d{2})(\d{2})')
    m = pattern.match(date_str)
    if m:
        year = m.group(1)
        month = m.group(2)
        day = m.group(3)
        return f"{year}-{month}-{day}"
    else:
        # 正規表現に合致しない場合はタイムスタンプの最初の年月日
        return "1970-01-01"


def conv_transaction_df2html(df: pd.DataFrame) -> list:
    """
    取引履歴のデータフレームを HTML のテーブルに変換し、リストにして返す
    :param df:
    :return:
    """
    list_html = list()
    list_html.append('<table class="simple">')

    list_html.append("<thead>")
    list_html.append("<tr>")
    for colname in df.columns:
        list_html.append(f"<th>{colname}</th>")
    list_html.append("</tr>")
    list_html.append("</thead>")

    list_html.append("<tbody>")
    for rowname in df.index:
        list_html.append("<tr>")
        for colname in df.columns:
            cell = df.at[rowname, colname]
            if pd.isna(cell):
                cell = ""
            match colname:
                case "注文番号":
                    list_html.append(f'<td style="text-align: right;">{cell}</td>')
                case "注文日時":
                    list_html.append(f'<td style="text-align: center;">{cell}</td>')
                case "銘柄コード":
                    list_html.append(f'<td style="text-align: center;">{cell}</td>')
                case "売買":
                    list_html.append(f'<td style="text-align: center;">{cell}</td>')
                case "約定単価":
                    list_html.append(f'<td style="text-align: right;">{cell}</td>')
                case "約定数量":
                    list_html.append(f'<td style="text-align: right;">{cell}</td>')
                case "損益":
                    list_html.append(f'<td style="text-align: right;">{cell}</td>')
                case "備考":
                    list_html.append(f'<td style="text-align: left;">{cell}</td>')
        list_html.append("</tr>")
    # 合計損益
    total = df["損益"].sum()
    list_html.append("<tr>")
    # 注文番号
    list_html.append('<td style="text-align: right;"></td>')
    # 注文日時
    list_html.append('<td style="text-align: center;"></td>')
    # 銘柄コード
    list_html.append('<td style="text-align: center;"></td>')
    # 売買
    list_html.append('<td style="text-align: center;"></td>')
    # 約定単価, 約定数量
    list_html.append('<td style="text-align: right;" colspan="2">合計損益</td>')
    # 損益
    list_html.append(f'<td style="text-align: right;">{total}</td>')
    # 備考"
    list_html.append('<td style="text-align: left;"></td>')
    list_html.append("</tr>")

    list_html.append("</tbody>")
    list_html.append("</table>")

    return list_html


def get_code_as_string(val) -> str:
    """
    東証の銘柄コードを文字列に
    :param val:
    :return:
    """
    if type(val) is str:
        ticker = val
    else:
        ticker = f"{int(val)}"
    return ticker


def min_max_scale(data):
    """
    Min-Maxスケーリングを使ってデータを [0, 1] の範囲に規格化する関数。
    :param data: 規格化したい数値データのリストまたはNumPy配列
    :return: [0, 1] の範囲に規格化されたデータ (numpy.ndarray)
    """
    data_array = np.array(data)

    # データの最小値と最大値を計算
    min_val = np.min(data_array)
    max_val = np.max(data_array)

    # 最小値と最大値が同じ場合（データがすべて同じ値の場合）は、ゼロ除算を避ける
    if max_val == min_val:
        return np.zeros_like(data_array)  # 全て 0 にするか、適切な値を返す

    # Min-Maxスケーリングを適用
    scaled_data = (data_array - min_val) / (max_val - min_val)

    return scaled_data
