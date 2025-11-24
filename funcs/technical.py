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


def calc_ma(df: pd.DataFrame, period_1: int = 60, period_2: int = 600) -> tuple[str, str]:
    """
    移動平均差
    :param df:
    :param period_1:
    :param period_2:
    :return:
    """
    colname1 = f"MA{period_1:03d}"
    colname2 = f"MA{period_2:03d}"
    df[colname1] = df["Price"].rolling(period_1, min_periods=1).mean()
    df[colname2] = df["Price"].rolling(period_2, min_periods=1).mean()
    return colname1, colname2

def calc_msd(df: pd.DataFrame, period: int = 60) -> str:
    """
    移動平均差
    :param df:
    :param period:
    :return:
    """
    colname = f"MSD{period:03d}"
    df[colname] = df["Price"].rolling(period, min_periods=1).std()
    return colname
