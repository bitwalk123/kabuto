import os

import pandas as pd

from funcs.ios import get_excel_sheet
from modules.rl_ppo_lite_20250825 import TradingSimulation
from structs.res import AppRes

if __name__ == "__main__":
    res = AppRes()
    code = "7011"
    # 学習対象のティックデータファイル・リスト: Time, Price, Volume の 3 列
    list_excel = [
        "tick_20250827.xlsx",
        "tick_20250828.xlsx",
    ]
    # 学習曲線用データフレーム
    df_lc = pd.DataFrame({
        "Epoch": list(),
        "Repeat": list(),
        "Data": list(),
        "Profit": list(),
    })
    df_lc = df_lc.astype(object)

    # シミュレータ・インスタンス
    model_path = os.path.join(res.dir_training, f"ppo_{code}_20250825.pth")
    sim = TradingSimulation(model_path)

    # 繰り返し学習回数（ティックデータ・リスト全体に亙って）
    repeats = 50

    epoch = 0
    # 繰り返し学習
    for repeat in range(repeats):
        for excel_file in list_excel:
            # ティックデータを読み込む
            path_excel = os.path.join(res.dir_excel, excel_file)
            df = get_excel_sheet(path_excel, code)  # "Time", "Price", "Volume" 列がある想定

            # 1行ずつシミュレーションに流す
            for i, row in df.iterrows():
                ts = row["Time"]
                price = row["Price"]
                volume = row["Volume"]
                # 最後の行だけ強制返済フラグを立てる
                force_close = (i == len(df) - 1)
                action = sim.add(ts, price, volume, force_close=force_close)

            # 結果（総収益）を保存
            df_result = sim.finalize()
            profit = df_result["Profit"].sum()
            print(f"Epoch: {epoch}, Repeat: {repeat}, File: {excel_file}, Total: {profit}")
            df_result.to_csv(
                os.path.join(res.dir_output, f"trade_results_{epoch:03}.csv")
            )

            # for plot of learning curve
            df_lc.at[epoch, "Epoch"] = epoch
            df_lc.at[epoch, "Repeat"] = repeat
            df_lc.at[epoch, "Data"] = excel_file
            df_lc.at[epoch, "Profit"] = profit

            epoch += 1

    # 学習曲線用データフレームの保存
    df_lc.to_csv(
        os.path.join(res.dir_output, f"learning_curve.csv")
    )
