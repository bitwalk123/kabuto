import datetime
import os
from typing import Any

import mplfinance as mpf
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import (
    font_manager as fm,
    pyplot as plt,
    dates as mdates,
    ticker as ticker,
)
from scipy.interpolate import griddata

from funcs.tide import get_format_date_from_date_str
from funcs.tse import get_ticker_name_list


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


def plot_main_effect(df, list_col, target, output):
    """
    主効果用プロット作成関数
    :param df:
    :param list_col:
    :param target:
    :param output:
    :return:
    """
    y_min_global = 1e7
    y_max_global = -1e7
    n = len(list_col)
    fig, ax = plt.subplots(n, 1, figsize=(6, 4 * n))
    for i, col in enumerate(list_col):
        sns.pointplot(x=col, y=target, data=df, ax=ax[i], errorbar="ci")
        y_min, y_max = ax[i].get_ylim()
        if y_min < y_min_global:
            y_min_global = y_min
        if y_max_global < y_max:
            y_max_global = y_max
        ax[i].grid()
        ax[i].set_title(f"{target}: {col}")

    for i in range(n):
        ax[i].set_ylim(y_min_global, y_max_global)

    plt.tight_layout()
    plt.savefig(output)
    plt.show()


def plot_interaction(df, pairs, target, output):
    """
    ### 交互作用プロット作成関数
    :param df:
    :param pairs:
    :param target:
    :param output:
    :return:
    """
    n = len(pairs)
    if n > 1:
        fig, ax = plt.subplots(len(pairs), 1, figsize=(6, n * 4))
        y_min_all = 1e6
        y_max_all = -1e6
        for i, (a, b) in enumerate(pairs):
            sns.pointplot(
                x=a,
                y=target,
                hue=b,
                data=df,
                markersize=6,
                linewidth=1,
                errorbar=None,
                palette="Set2",
                ax=ax[i],
            )
            ax[i].set_title(f"{target}: {a} × {b}")
            ax[i].set_xlabel(a)
            ax[i].set_ylabel(target)
            # 凡例のタイトル
            lg = ax[i].legend(fontsize=7)
            lg.set_title(b, prop={"size": 7})

            ax[i].grid()

            # y 軸の最大値・最小値
            y_min, y_max = ax[i].get_ylim()
            if y_min < y_min_all:
                y_min_all = y_min
            if y_max_all < y_max:
                y_max_all = y_max

        # y 軸の範囲を揃える
        for i in range(n):
            ax[i].set_ylim(y_min_all, y_max_all)
    else:
        a, b = pairs[0]
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        sns.pointplot(
            x=a,
            y=target,
            hue=b,
            data=df,
            markersize=6,
            linewidth=1,
            errorbar=None,
            palette="Set2",
            ax=ax,
        )
        ax.set_title(f"{target}: {a} × {b}")
        ax.set_xlabel(a)
        ax.set_ylabel(target)
        # 凡例のタイトル
        lg = ax.legend(fontsize=7)
        lg.set_title(b, prop={"size": 7})
        ax.grid()

    plt.tight_layout()
    plt.savefig(output)
    plt.show()


def plot_contour(df, col_x: str, col_y: str, col_z: str, output: str):
    x = df[col_x]
    y = df[col_y]
    z = df[col_z]

    # グリッド作成
    xi = np.linspace(x.min(), x.max(), 50)
    yi = np.linspace(y.min(), y.max(), 50)
    Xi, Yi = np.meshgrid(xi, yi)
    # 補間
    Zi = griddata((x, y), z, (Xi, Yi), method="cubic")

    fig, ax = plt.subplots(figsize=(6, 6))

    cont = ax.contour(Xi, Yi, Zi, levels=15, cmap="coolwarm")
    ax.clabel(cont, inline=True, fontsize=12)
    # ✅ 元データ点を黒丸で追加
    ax.scatter(x, y, color="black", s=5, zorder=3)

    y_min, y_max = ax.get_ylim()
    if y_max < 10:
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:4,.1f}"))
    else:
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:4,.0f}"))
    ax.set_title(f"{col_z}: {col_x} × {col_y}")
    ax.set_xlabel(col_x)
    ax.set_ylabel(col_y)

    ax.grid(True, color="gray", linestyle="dotted", linewidth=0.5)
    plt.savefig(output)
    plt.show()


def trend_label_html(text: str, size: int = 9) -> str:
    return f'<span style="font-size: {size}pt; font-family: monospace;">{text}</span>'


def plot_trend_review(
        code: str,
        df: pd.DataFrame,
        target_dir: str,
        dict_ts: dict[str, datetime.datetime],
        dict_setting: dict[str, Any],
        date_str: str
) -> None:
    """
    レビュー用チャートの作成と保存（Jupyter 用）
    :param code:
    :param df:
    :param target_dir:
    :param dict_ts:
    :param dict_setting:
    :param date_str:
    :return:
    """
    n = 4

    # Matplotlib の共通設定
    FONT_PATH = "fonts/RictyDiminished-Regular.ttf"
    fm.fontManager.addfont(FONT_PATH)

    # FontPropertiesオブジェクト生成（名前の取得のため）
    font_prop = fm.FontProperties(fname=FONT_PATH)
    font_prop.get_name()

    plt.rcParams["font.family"] = font_prop.get_name()

    if n == 1:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.grid(axis="y")
    else:
        fig = plt.figure(figsize=(6, 6))
        ax = dict()
        gs = fig.add_gridspec(
            n,
            1,
            wspace=0.0,
            hspace=0.0,
            height_ratios=[2 if i <= 1 else 1 for i in range(n)],
        )
        for i, axis in enumerate(gs.subplots(sharex="col")):
            ax[i] = axis
            ax[i].grid(axis="y")

    name = get_ticker_name_list([code])[code]
    format_date = get_format_date_from_date_str(date_str)
    ax[0].plot(df["price"], linewidth=0.5, color="lightgray", alpha=0.5, label="株価")
    ax[0].plot(df["ma1"], linewidth=0.75, color="#008000", label="移動平均線 MA1")
    ax[0].plot(df["vwap"], linewidth=0.75, color="#ff00ff", label="VWAP")

    td = datetime.timedelta(minutes=15)
    ax[0].set_xlim(dict_ts["start"] - td, dict_ts["end"] + td)

    ax[0].set_ylabel("株価")
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax[0].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    ax[0].legend(fontsize=6)

    ax[1].plot(df["price"] - df["vwap"], linewidth=0.5, color="lightgray", alpha=0.5, label="株価 - VWAP", )
    ax[1].plot(df["ma1"] - df["vwap"], linewidth=0.75, color="#c06000", label="MA1 - VWAP")

    ax[1].axhline(y=0, linewidth=0.5, color="black")
    ax[1].set_ylabel("乖離度")
    ax[1].legend(fontsize=6)

    ax[2].plot(df["profit"], linewidth=0.1, color="#ff00ff", label="含み損益")
    ax[2].fill_between(df.index, df["profit"], color="#ff00ff", alpha=0.15)
    ax[2].plot(df["profit_max"], linewidth=0.75, color="#c00000", label="最大含み損益")
    ax[2].axhline(y=dict_setting["DD_PROFIT"], linewidth=0.75, color="C0", alpha=1, label="トレーリング")
    ax[2].set_ylabel("含み損益")
    ax[2].legend(fontsize=6)

    ax[3].plot(df["dd_ratio"], linewidth=0.5, color="C0", alpha=0.75, label="DD ratio")
    ax[3].axhline(
        y=dict_setting["DD_RATIO"], linewidth=0.75, color="C1", label="利確ライン"
    )
    ax[3].set_ylabel("DD ratio")
    ax[3].legend(fontsize=6)
    ax[3].set_ylim(0, 1.1)

    list_cross = df[df["cross1"] != 0].index
    print(f"# of cross: {len(list_cross)}")
    for i in range(n):
        for t in list_cross:
            cname = "#f00000" if 0 < df.at[t, "cross1"] else "#0000d0"
            ax[i].axvline(x=t, color=cname, linestyle="solid", alpha=0.25, linewidth=0.75)

        x = [dict_ts["start"], dict_ts["trade"]]
        ax[i].fill_between(x, 0, 1, color="black", alpha=0.15, transform=ax[i].get_xaxis_transform())

    ax[0].set_title(f"{format_date}: {name} ({code})")
    ax[n - 1].set_xlabel(f"# of crossed: {len(list_cross)} times")

    # plt.suptitle(title_str, fontsize=5)
    plt.tight_layout()
    # plt.subplots_adjust(top=0.89)
    output = os.path.join(target_dir, f"{code}_trend_technical.png")
    print(output)
    plt.savefig(output)
    plt.show()


def trend_diff(code: str, df: pd.DataFrame):
    # 出力イメージ名
    dt_end = df.tail(1).index[0].date()
    str_year = f"{dt_end.year:04d}"
    str_month = f"{dt_end.month:02d}"
    str_day = f"{dt_end.day:02d}"

    dir_name = os.path.join(str_year, str_month, str_day)
    os.makedirs(dir_name, exist_ok=True)
    img_name = os.path.join(dir_name, f"{code}_trend_diff.png")

    n = len(df)
    mean = df["Diff"].mean()
    std = df["Diff"].std()
    median = df["Diff"].median()
    iqr = df["Diff"].quantile(0.75) - df["Diff"].quantile(0.25)
    footer = f"High - Low: n={n} / mean={mean:.1f}, stdev={std:.1f} / median={median:.1f}, IQR={iqr:.1f}"

    FONT_PATH = "../fonts/RictyDiminished-Regular.ttf"
    fm.fontManager.addfont(FONT_PATH)

    # FontPropertiesオブジェクト生成（名前の取得のため）
    font_prop = fm.FontProperties(fname=FONT_PATH)
    font_prop.get_name()

    plt.rcParams["font.family"] = font_prop.get_name()
    plt.rcParams["font.size"] = 9
    n = 2

    fig = plt.figure(figsize=(6, 3))
    ax = dict()
    gs = fig.add_gridspec(
        n,
        1,
        wspace=0.0,
        hspace=0.0,
        height_ratios=[1.5 if i <= 0 else 1 for i in range(n)],
    )
    for i, axis in enumerate(gs.subplots(sharex="col")):
        ax[i] = axis
        ax[i].grid()

    name = get_ticker_name_list([code])[code]
    ax[0].set_title(f"{name} ({code})")

    apds = [
        mpf.make_addplot(df["Diff"], width=0.75, color="C1", ax=ax[1]),
    ]
    mpf.plot(
        df,
        type="candle",
        style="default",
        addplot=apds,
        datetime_format="%m/%d",
        xrotation=0,
        update_width_config=dict(candle_linewidth=0.75),
        ax=ax[0],
    )
    ax[0].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))

    ax[1].set_xlabel(footer)
    ax[1].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    _, high = ax[1].get_ylim()
    ax[1].set_ylim(0, high)
    ax[1].set_ylabel("High - Low")

    plt.tight_layout()
    plt.savefig(img_name)
    plt.show()
