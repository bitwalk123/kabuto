import xlwings as xw  # 1. xlwingsをインポート

wb = xw.Book('VBA_example.xlsm')  # 2. ブックを開く
macro = wb.macro('hello_VBA')  # 3. マクロを取得
macro()  # 4. マクロを実行
