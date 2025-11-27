import pandas as pd
from matplotlib import (
    font_manager as fm,
    pyplot as plt,
    dates as mdates,
)


def plot_mpl_chart(df: pd.DataFrame, title: str, condition: str, imgname: str):
    FONT_PATH = "fonts/RictyDiminished-Regular.ttf"
    fm.fontManager.addfont(FONT_PATH)

    # FontPropertiesオブジェクト生成（名前の取得のため）
    font_prop = fm.FontProperties(fname=FONT_PATH)
    font_prop.get_name()

    plt.rcParams["font.family"] = font_prop.get_name()
    plt.rcParams["font.size"] = 16

    fig = plt.figure(figsize=(12, 8))
    ax = dict()
    n = 2
    gs = fig.add_gridspec(
        n, 1, wspace=0.0, hspace=0.0,
        height_ratios=[3 if i == 0 else 1 for i in range(n)]
    )
    for i, axis in enumerate(gs.subplots(sharex="col")):
        ax[i] = axis
        ax[i].grid()

    ax[0].plot(df["MinMaxPrice"], color="gray", linewidth=0.5)

    # Parabolic SAR
    ser_bull = df[df["Trend"] > 0]["PSAR"]
    ser_bear = df[df["Trend"] < 0]["PSAR"]
    ax[0].scatter(x=ser_bull.index, y=ser_bull, s=5, c="red")
    ax[0].scatter(x=ser_bear.index, y=ser_bear, s=5, c="blue")

    ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax[0].set_ylabel("Normalized Price")
    ax[0].set_title(condition, fontsize="small")

    ax[1].plot(df["EPupd"], color="#880", linewidth=0.5)
    ax[1].set_ylabel("EP update")

    plt.suptitle(title)
    plt.savefig(imgname)
    plt.close()


def plot_obs_trend(df: pd.DataFrame, title: str = ""):
    FONT_PATH = "fonts/RictyDiminished-Regular.ttf"
    fm.fontManager.addfont(FONT_PATH)

    # FontPropertiesオブジェクト生成（名前の取得のため）
    font_prop = fm.FontProperties(fname=FONT_PATH)
    font_prop.get_name()

    plt.rcParams["font.family"] = font_prop.get_name()
    plt.rcParams["font.size"] = 9

    n = len(df.columns)
    fig = plt.figure(figsize=(18, 0.2 + n))
    gs = fig.add_gridspec(
        n, 1,
        wspace=0.0, hspace=0.0,
        height_ratios=[1 for r in range(n)]
    )
    ax = dict()
    for i, axis in enumerate(gs.subplots(sharex="col")):
        ax[i] = axis
        ax[i].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax[i].grid()

    for i, colname in enumerate(df.columns):
        if colname == "低ボラ":
            x = df.index
            y = df[colname]
            # ax[i].plot(df[colname], color='green', alpha=0.5, linewidth=0.5)
            # ax[i].fill_between(x, 0, 1, where=y == 1.0, color='green', alpha=0.5, transform=ax[i].get_xaxis_transform())
            ax[i].fill_between(x, 0, y, where=y == 1.0, color='green', alpha=0.5, interpolate=True)
        else:
            ax[i].plot(df[colname], linewidth=0.5)

        ax[i].set_ylabel(colname)
    ax[0].set_title(title)

    plt.tight_layout()
    plt.show()
