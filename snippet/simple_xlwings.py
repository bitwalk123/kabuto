import xlwings as xw

if __name__ == "__main__":
    wb = xw.Book()  # 新しいワークブックを開く
    sheet = wb.sheets["Sheet1"] # シート・オブジェクトをインスタンス化
    sheet["A1"].value = "Python から書き込みました。" # 値の書き込み
