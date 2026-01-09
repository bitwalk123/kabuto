import os

from funcs.ios import get_excel_sheet
from funcs.logs import setup_logging
from funcs.setting import load_setting
from funcs.tide import get_intraday_timestamp
from structs.res import AppRes
from widgets.explorer import Explorer

if __name__ == "__main__":
    main_logger = setup_logging()
    res = AppRes()

    code = "7011"
    # パラメータ設定ファイルから読み込む
    dict_setting_base = load_setting(res, code)

    excel = "ticks_20260108.xlsx"

    # ザラ場の開始時間などのタイムスタンプ取得（Excelの日付）
    dict_ts = get_intraday_timestamp(excel)
    # メイン・インスタンスの生成
    app = Explorer(code, dict_ts)

    # 指定した Excel ファイル、銘柄コードから、ティックデータをデータフレームに読み込む
    path_excel = os.path.join(res.dir_collection, excel)
    df = get_excel_sheet(path_excel, code)

    dict_setting = dict_setting_base

    # パラメータ設定とティックデータでシミュレーションを実施
    app.run(dict_setting, df)

    # 取引明細
    df_transaction = app.getTransaction()
    print("\n取引明細")
    print(df_transaction)
    n_trade = len(df_transaction)
    total = df_transaction["損益"].sum()
    print(f"取引回数: {n_trade} 回 / 総収益: {total} 円/株")

    # メイン・インスタンスの削除
    app = None
