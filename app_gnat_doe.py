import glob
import os
import sys

import pandas as pd

from funcs.excel import get_excel_sheet
from modules.agent import SimulationAgent

if __name__ == "__main__":
    name_doe: str = "doe-000"
    code: str = "9984"

    dict_setting: dict = {}
    try:
        df_doe = pd.read_csv(os.path.join("doe", name_doe, "doe.csv"))  # DOE条件のCSVファイルを読み込み
    except FileNotFoundError:
        print(f"DOE条件のCSVファイルが見つかりません: doe/{name_doe}/doe.csv")
        sys.exit(1)

    # 学習に使用するティックデータ
    home = os.path.expanduser("~")
    dir_excel = os.path.join(home, "MyProjects", "kabuto", "collection")
    path_excel = os.path.join(dir_excel, "*.xlsx")
    list_file_excel = sorted(glob.glob(path_excel))[-1:]

    print("集計するファイル")
    for file_excel in list_file_excel:
        print(file_excel)

    df_result = pd.DataFrame()
    for file_excel in list_file_excel:
        df_excel = get_excel_sheet(file_excel, code)
        for r in range(len(df_doe)):
            row = df_doe.iloc[r]
            r2 = len(df_result)
            for colname in df_doe.columns:
                dict_setting[colname] = row[colname]
                df_result.loc[r2, colname] = row[colname]
            print("### 条　　件 ###")
            for key, value in dict_setting.items():
                print(key, ":", value)

            # ループ毎に新しいインスタンスを生成
            agent = SimulationAgent(code, dict_setting)
            df_technicals, df_transaction = agent.run(df_excel)

            print(df_transaction)
            n_transaction = len(df_transaction)
            pnl = df_transaction["損益"].sum()
            print(f"約定係数: {n_transaction} 回, 損益: {pnl} 円/株")

    """
    df_result = pd.concat([df_result_pre, df_result], ignore_index=True)
    print(df_result)
    df_result.to_csv(csv_result, index=False)
    """
