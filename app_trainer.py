import os

import pandas as pd

from funcs.ios import get_excel_sheet
from structs.res import AppRes

if __name__ == "__main__":
    res = AppRes()
    code = "7011"
    list_excel = [
        "tick_20250819.xlsx",
        "tick_20250820.xlsx",
        "tick_20250821.xlsx",
        "tick_20250822.xlsx",
        "tick_20250825.xlsx",
        "tick_20250826.xlsx",
        "tick_20250827.xlsx",
        "tick_20250828.xlsx",
        "tick_20250829.xlsx",
        "tick_20250901.xlsx",
    ]
    # 学習曲線用データフレーム
    df_lc = pd.DataFrame({
        "Epoch": list(),
        "Data": list(),
        "Profit": list(),
    })
    df_lc = df_lc.astype(object)

    epoch = 0
    for n in range(10):
        for file_excel in list_excel:
            df = get_excel_sheet(os.path.join(res.dir_excel, file_excel), code)

            trainer = Trainer()
            df_transaction = trainer.train(df)
            profit = df_transaction["Profit"].sum()
            print(f"Epoch: {epoch}, {file_excel}, 総収益: {profit}")
            df_transaction.to_csv(
                os.path.join(res.dir_output, f"trade_results_{epoch:03}.csv")
            )
            # for plot of learning curve
            df_lc.at[epoch, "Epoch"] = epoch
            df_lc.at[epoch, "Data"] = file_excel
            df_lc.at[epoch, "Profit"] = profit

            epoch += 1

    # 学習曲線用データフレームの保存
    df_lc.to_csv(
        os.path.join(res.dir_output, f"learning_curve.csv")
    )
