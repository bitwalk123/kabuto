import numpy as np
import pandas as pd


def conv_transaction_df2html(df: pd.DataFrame) -> list:
    """
    取引履歴のデータフレームを HTML のテーブルに変換し、リストにして返す
    :param df:
    :return:
    """
    list_html = list()
    list_html.append("<table>")

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


def min_max_scale(data):
    """
    Min-Maxスケーリングを使ってデータを [0, 1] の範囲に規格化する関数。

    Args:
        data (array-like): 規格化したい数値データのリストまたはNumPy配列。

    Returns:
        numpy.ndarray: [0, 1] の範囲に規格化されたデータ。
    """
    data_array = np.array(data)

    # データの最小値と最大値を計算
    min_val = np.min(data_array)
    max_val = np.max(data_array)

    # 最小値と最大値が同じ場合（データがすべて同じ値の場合）は、ゼロ除算を避ける
    if max_val == min_val:
        return np.zeros_like(data_array)  # 全て0にするか、適切な値を返す

    # Min-Maxスケーリングを適用
    scaled_data = (data_array - min_val) / (max_val - min_val)

    return scaled_data
