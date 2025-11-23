import pandas as pd


def calc_vwap(df: pd.DataFrame):
    """
    ティックデータから VWAP を一括計算
    :param df:
    :return:
    """
    # 差分出来高を算出（最初の行は初期出来高、2 行目以降は増加分）
    df["vol_diff"] = df["Volume"].diff().fillna(df["Volume"])
    # VWAP の計算
    df["cum_pv"] = (df["Price"] * df["vol_diff"]).cumsum()
    df["cum_vol"] = df["vol_diff"].cumsum()
    df["VWAP"] = df["cum_pv"] / df["cum_vol"]
    # VWAP乖離率の計算
    df["VWAP_deviation"] = (df["Price"] - df["VWAP"]) / df["VWAP"]


def calc_ma(df: pd.DataFrame, t1: int = 60, t2: int = 600) -> tuple[str, str]:
    """
    移動平均差
    :param df:
    :param t1:
    :param t2:
    :return:
    """
    colname1 = f"MA{t1:03d}"
    colname2 = f"MA{t2:03d}"
    df[colname1] = df["Price"].rolling(t1, min_periods=1).mean()
    df[colname2] = df["Price"].rolling(t2, min_periods=1).mean()
    return colname1, colname2
