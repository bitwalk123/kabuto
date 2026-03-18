import datetime
import os
from typing import Any

import mplfinance as mpf
import numpy as np
import pandas as pd
import seaborn as sns
import talib as ta
from matplotlib import dates as mdates
from matplotlib import font_manager as fm
from matplotlib import pyplot as plt
from matplotlib import ticker as ticker
from pandas import DataFrame
from scipy.interpolate import griddata

from funcs.tide import get_format_date_from_date_str
from funcs.tse import get_ticker_name_list
from modules.technical import RSI, Momentum


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

    # 銘柄情報
    name = get_ticker_name_list([code])[code]
    # 今日の High - Low
    price_high = df.tail(1)["High"].iloc[0]
    price_low = df.tail(1)["Low"].iloc[0]
    price_delta = price_high - price_low
    today = df.tail(1).index[0].date()
    ax[0].set_title(f"{name} ({code})\n{today}: High - Low = {price_delta:.1f} JPY")

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


def mpl_plot_review(
        df: pd.DataFrame,
        title: str,
        dict_ts: dict[str, Any],
        dict_setting: dict[str, Any]
) -> plt.Figure:
    # Matplotlib の共通設定
    FONT_PATH = "fonts/RictyDiminished-Regular.ttf"
    fm.fontManager.addfont(FONT_PATH)

    # FontPropertiesオブジェクト生成（名前の取得のため）
    font_prop = fm.FontProperties(fname=FONT_PATH)
    font_prop.get_name()

    plt.rcParams["font.family"] = font_prop.get_name()
    plt.rcParams["font.size"] = 9

    n = 3
    fig = plt.figure(figsize=(6, 6))
    ax = dict()
    gs = fig.add_gridspec(
        n,
        1,
        wspace=0.0,
        hspace=0.0,
        height_ratios=[1.5 if i == 0 else 1 for i in range(n)],
    )
    for i, axis in enumerate(gs.subplots(sharex="col")):
        ax[i] = axis
        ax[i].grid(axis="y")

    # 株価と VWAP
    plot_price_vwap(ax[0], df, title, dict_ts)

    # 含み益
    plot_profit(ax[1], df, dict_setting)

    # ドローダウン
    ax2 = ax[2]
    plot_drawdown(ax2, df, dict_setting)

    # クロス・シグナル、その他縦線系
    plot_verticals(n, ax, df, dict_ts)

    fig.tight_layout()
    return fig


def plot_price_vwap(ax: plt.Axes, df: DataFrame, title: str, dict_ts: dict[str, Any]):
    ax.set_title(title)
    td = datetime.timedelta(minutes=15)
    ax.set_xlim(dict_ts["start"] - td, dict_ts["end"] + td)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    # 株価と VWAP
    ax.plot(df["price"], linewidth=0.5, color="black", alpha=0.5, label="株価")
    ax.plot(df["ma1"], linewidth=0.75, color="#0c0", label="移動平均線 MA1")
    # 評価用の移動平均 MA2
    df["ma2"] = df["price"].rolling(300, min_periods=1).mean()
    ax.plot(df["ma2"], linewidth=0.5, color="#00c", label="移動平均線 MA2")
    ax.plot(df["vwap"], linewidth=0.75, color="#f0f", label="VWAP")

    ax.set_ylabel("株価")
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left", borderaxespad=0.5, fontsize=6)


def plot_momentum(ax: plt.Axes, df: DataFrame, dict_setting: dict[str, Any]):
    # モメンタム
    n = 150
    mom = Momentum(n)
    df["momentum"] = [mom.update(v) for v in df["ma1"]]
    ax.plot(df["momentum"], color="#888", linewidth=0.25, alpha=0.75, label=f"n = {n:d}")
    x = df.index
    y = df["momentum"]
    ax.fill_between(x, 0, y, where=(0 < y), fc="#faa", ec="#f00", alpha=0.5, lw=0.5)
    ax.fill_between(x, 0, y, where=(y < 0), fc="#aaf", ec="#00f", alpha=0.5, lw=0.5)

    ax.axhline(y=0, linewidth=0.5, color="black", alpha=0.5)

    ax.set_ylabel("モメンタム")
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left", borderaxespad=0.5, fontsize=6)


def plot_rsi(ax: plt.Axes, df: DataFrame, dict_setting: dict[str, Any]):
    ax.plot(df["rsi"], color="#888", linewidth=0.25, alpha=0.75, label=f"n = {dict_setting["PERIOD_RSI"]}")
    x = df.index
    y = df["rsi"]
    ax.fill_between(x, 0.5, y, where=(0.5 < y), fc="#faa", ec="#f00", alpha=0.5, lw=0.5)
    ax.fill_between(x, 0.5, y, where=(y < 0.5), fc="#aaf", ec="#00f", alpha=0.5, lw=0.5)

    ax.axhline(y=0.5, linewidth=0.75, color="black", alpha=0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel(f"RSI")
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left", borderaxespad=0.5, fontsize=6)


def plot_profit(ax: plt.Axes, df: DataFrame, dict_setting: dict[str, Any]):
    # 含み益
    x = df.index
    y1 = df["profit"]
    y2 = df["profit_max"]
    y_dd_th = dict_setting["DD_PROFIT"]

    ax.fill_between(x, 0, y1, where=(0 < y1), fc="#fcc", ec="#f00", alpha=0.5, lw=0.5, label="含み益")
    ax.fill_between(x, 0, y1, where=(y1 < 0), fc="#ccf", ec="#00f", alpha=0.5, lw=0.5, label="含み損")

    ax.plot(y2, linewidth=0.75, color="#a00", label="最大含み損益")
    ax.axhline(y=y_dd_th, linewidth=0.75, color="C0", alpha=1, label="トレーリング")

    ax.set_ylabel("含み損益")
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left", borderaxespad=0.5, fontsize=6)


def plot_drawdown(ax: plt.Axes, df: DataFrame, dict_setting: dict[str, Any]):
    # ドローダウン
    y_dd_th = dict_setting["DD_PROFIT"]

    # 最大含み益がしきい値を超えているときのみ、ドラーダウンの比率を表示
    df["dd_ratio_2"] = [
        k2 if y_dd_th <= k1 else 0 for k1, k2 in zip(df["profit"], df["dd_ratio"])
    ]
    y_ddr_1 = df["dd_ratio_2"]
    y_ddr_2 = dict_setting["DD_RATIO"]

    ax.plot(y_ddr_1, linewidth=0.75, color="C0", alpha=0.75, label="DD ratio")
    ax.axhline(y=y_ddr_2, linewidth=0.75, color="C1", label="利確ライン")

    ax.set_ylim(0, y_ddr_2 + 0.1)
    ax.set_ylabel("DD ratio")
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left", borderaxespad=0.5, fontsize=6)


def plot_verticals(
        n: int,
        ax: dict[int, plt.Axes],
        df: DataFrame,
        dict_setting: dict[str, Any],
        dict_ts: dict[str, Any],
):
    # クロス・シグナル、ウォームアップ期間
    list_cross = df[df["cross1"] != 0].index
    ax[n - 1].set_xlabel(f"# of crossed: {len(list_cross)} times")
    for i in range(n):
        # クロス・シグナル
        for t in list_cross:
            cname = "#f00" if 0 < df.at[t, "cross1"] else "#00f"
            ax[i].axvline(x=t, c=cname, ls="solid", alpha=0.25, lw=0.75)

        # ウォークアップ期間
        td = datetime.timedelta(seconds=dict_setting["PERIOD_WARMUP"] * 2)
        x = [dict_ts["start"], df.index[0] + td]
        ax[i].fill_between(
            x, 0, 1, color="black", alpha=0.15, transform=ax[i].get_xaxis_transform()
        )
