import datetime

from matplotlib import dates as mdates
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd

from funcs.commons import get_collection_path
from funcs.ios import get_excel_sheet
from modules.env import TradingEnv
from structs.res import AppRes


def plot_obs_trend(df: pd.DataFrame, title: str):
    FONT_PATH = "fonts/RictyDiminished-Regular.ttf"
    fm.fontManager.addfont(FONT_PATH)

    # FontPropertiesオブジェクト生成（名前の取得のため）
    font_prop = fm.FontProperties(fname=FONT_PATH)
    font_prop.get_name()

    plt.rcParams["font.family"] = font_prop.get_name()
    plt.rcParams["font.size"] = 9

    n = len(df.columns)
    fig = plt.figure(figsize=(6, 0.2 + n))
    ax = dict()
    gs = fig.add_gridspec(
        n, 1,
        wspace=0.0, hspace=0.0,
        height_ratios=[1 for i in range(n)]
    )
    for i, axis in enumerate(gs.subplots(sharex="col")):
        ax[i] = axis
        ax[i].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax[i].grid()

    for i, colname in enumerate(df.columns):
        ax[i].plot(df[colname], linewidth=0.5)
        y_min, y_max = ax[i].get_ylim()
        if -1.1 < y_min:
            y_min = -1.1
        if y_max < 1.1:
            y_max = 1.1
        ax[i].set_ylim(y_min, y_max)
        if colname == "RSI":
            ax[i].axhline(0.6, linewidth=0.5, color="C1")
            ax[i].axhline(-0.6, linewidth=0.5, color="C1")
        ax[i].set_ylabel(colname)
    ax[0].set_title(title)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    res = AppRes()
    env = TradingEnv()
    # タイムスタンプへ時差を加算・減算用（Asia/Tokyo)
    tz = 9. * 60 * 60

    # 推論用データ
    # file = "ticks_20251006.xlsx"
    file = "ticks_20251118.xlsx"
    code = "7011"

    print(f"過去データ {file} の銘柄 {code} について観測値を算出します。")
    # Excel ファイルのフルパス
    path_excel = get_collection_path(res, file)
    # Excel ファイルをデータフレームに読み込む
    df = get_excel_sheet(path_excel, code)
    list_obs = list()
    for row in df.index:
        ts = df.loc[row, "Time"]
        price = df.loc[row, "Price"]
        volume = df.loc[row, "Volume"]
        obs = env.getObservation(ts, price, volume)
        list_obs.append(obs)

    list_name = [
        "株価比",
        "MAΔ",
        "VWAPΔ",
        "Mσ",
        "含損益",
        "含損益M",
        "HOLD1",
        "HOLD2",
        "TRADE",
        "NONE",
        "LONG",
        "SHORT"
    ]
    df_obs = pd.concat([pd.Series(row) for row in list_obs], axis=1).T
    df_obs.columns = list_name
    list_dt = pd.to_datetime([datetime.datetime.fromtimestamp(ts) for ts in df["Time"]])
    df_obs.index = list_dt[:len(df_obs)]
    # print(df_obs)
    list_col = list_name[:4]
    df_obs_sub = df_obs[list_col]
    title = f"Observation trends from tick data\n{file} / {code}"
    plot_obs_trend(df_obs_sub, title)
    print(df_obs_sub.describe())
