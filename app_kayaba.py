"""
Project Kabuto のバックテスト用 CLI アプリの起動プログラム (Kayaba)
"""
import os
from typing import Any

import pandas as pd

from funcs.ios import load_excel
from funcs.logs import setup_logging
from funcs.setting import load_setting
from modules.kayaba import Kayaba
from structs.res import AppRes

if __name__ == "__main__":
    main_logger = setup_logging()
    res = AppRes()

    # 銘柄コード
    code = "9984"
    # パラメータの読み込み
    dict_setting: dict[str, Any] = load_setting(res, code)
    # ティックファイル
    date_str = "20260319"
    path_excel = os.path.join(res.dir_collection, f"ticks_{date_str}.xlsx")
    dict_sheet = load_excel(path_excel)
    df: pd.DataFrame = dict_sheet[code]

    # Kayaba インスタンス
    app = Kayaba(res, code, dict_setting, df)
    total = app.run()
    print(f"損益: {total: .0f} 円")

    # テクニカル・データ（出力先）
    path_csv = os.path.join(res.dir_temp, f"{date_str}_{code}_technicals.csv")
    app.saveTechnicals(path_csv)