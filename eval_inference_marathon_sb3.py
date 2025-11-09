import os
import sys
from time import perf_counter

import pandas as pd

from funcs.commons import get_collection_path
from funcs.ios import get_excel_sheet
from funcs.models import get_ppo_model_path
from modules.agent import PPOAgentSB3
from structs.res import AppRes

if __name__ == "__main__":
    res = AppRes()
    agent = PPOAgentSB3()

    dict_result = {
        "ファイル": [],
        "取引回数": [],
        "総損益": [],
    }

    # 推論用データ群
    list_file = sorted(os.listdir(res.dir_collection))
    code = "7011"

    # ループ開始時刻
    t_start = perf_counter()
    n_tick = 0
    for file in list_file:
        print(f"過去データ {file} の銘柄 {code} について推論します。")
        # Excel ファイルのフルパス
        path_excel = get_collection_path(res, file)
        # Excel ファイルをデータフレームに読み込む
        df = get_excel_sheet(path_excel, code)
        n_tick += len(df)
        path_model = get_ppo_model_path(res, code)

        result = agent.infer(df, path_model)
        if not result:
            sys.exit("正常終了しませんでした。")

        # 取引結果
        df_transaction: pd.DataFrame = agent.results["transaction"]
        n_transaction = len(df_transaction)
        pnl_total = df_transaction['損益'].sum()

        print(f"取引回数 : {n_transaction} 回, 一株当りの損益 : {pnl_total} 円")
        dict_result["ファイル"].append(file)
        dict_result["取引回数"].append(n_transaction)
        dict_result["総損益"].append(pnl_total)

    # ループ終了時刻
    t_end = perf_counter()

    print("----------------------------------------------------------------")
    df_result = pd.DataFrame(dict_result)
    print(df_result)
    print(f"総々収益 : {df_result["総損益"].sum()} 円")

    t_delta = t_end - t_start
    print(f"計測時間 :\t\t{t_delta * 1_000:,.6f} msec")
    print(f"総ティック量 :\t\t{n_tick:,d} tick")
    print(f"時間 / 1 ティック :\t{t_delta / n_tick * 1_000:.6f} msec")
