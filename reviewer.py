import os

import pandas as pd

from funcs.plot import plot_trend_review
from funcs.setting import load_setting
from funcs.tide import conv_date_str_to_path, conv_str_to_datetime
from structs.res import AppRes

if __name__ == "__main__":
    debug = False

    res = AppRes()
    date_str = "20260227"
    dict_ts = conv_str_to_datetime(date_str)

    list_ticker = ["4005", "5020", "8593"]
    # list_ticker = ["5020"]

    for code in list_ticker:
        dict_setting = load_setting(res, code)
        print(dict_setting)

        if debug:
            base_dir = res.dir_temp
        else:
            base_dir = res.dir_output

        target_dir = os.path.join(base_dir, conv_date_str_to_path(date_str))
        path_csv = os.path.join(target_dir, f"{code}_technicals.csv")
        df = pd.read_csv(path_csv, index_col=0)
        df.index = [pd.to_datetime(t) for t in df.index]

        print(df.columns)
        plot_trend_review(code, df, target_dir, dict_ts, dict_setting, date_str)