import numpy as np
import pandas as pd

from structs.res import AppRes


def get_ticker_list() -> pd.DataFrame:
    # JPX ticker list
    res = AppRes()
    df_tse = res.getJPXTickerList()
    market_target = ['グロース（内国株式）', 'スタンダード（内国株式）', 'プライム（内国株式）']
    df = df_tse[df_tse['市場・商品区分'].isin(market_target)].reset_index(drop=True)
    df['コード'] = df['コード'].astype(str)
    df['日時'] = None
    df['日時'] = df['日時'].astype(str)
    df['高値'] = np.nan
    df['安値'] = np.nan
    df['変化率'] = np.nan
    df['出来高'] = np.nan
    df['増減'] = None
    df['増減'] = df['増減'].astype(str)

    return df


def get_ticker_name_list(list_ticker: list) -> dict:
    df = get_ticker_list()
    list_name = list(df[df["コード"].isin(list_ticker)]["銘柄名"])
    dict_name = dict()
    for ticker, name in zip(list_ticker, list_name):
        dict_name[ticker] = name
    return dict_name
