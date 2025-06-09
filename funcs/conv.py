import pandas as pd


def conv_transaction_df2html(df: pd.DataFrame) -> list:
    list_html = list()
    list_html.append("<table>")

    list_html.append("<thead>")
    list_html.append("<tr>")
    for colname in df.columns:
        list_html.append(f"<th>{colname}</th>")
    list_html.append("</tr>")
    list_html.append("</thead>")

    list_html.append("<tbody")
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
