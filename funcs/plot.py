import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import (
    font_manager as fm,
    pyplot as plt,
    dates as mdates, ticker,
)
from scipy.interpolate import griddata


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
