import sys

from matplotlib import dates as mdates
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd

from funcs.commons import get_collection_path
from funcs.ios import get_excel_sheet
from funcs.models import get_ppo_model_path
from modules.agent import PPOAgentSB3
from structs.res import AppRes

FONT_PATH = "fonts/RictyDiminished-Regular.ttf"
fm.fontManager.addfont(FONT_PATH)

# FontPropertiesオブジェクト生成（名前の取得のため）
font_prop = fm.FontProperties(fname=FONT_PATH)
font_prop.get_name()

plt.rcParams["font.family"] = font_prop.get_name()


def plot_bar_profit(df: pd.DataFrame):
    df.index = pd.to_datetime(df["注文日時"])
    df.index.name = "DateTime"
    ser = df["損益"].dropna()
    total = int(ser.sum())
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.bar(ser.index, ser, width=0.0005, label=f"総収益 : {total:d} 円")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_ylabel("確定損益（円/株）")

    if len(df) > 0:
        dt = df.index[0]
        dt_start = pd.to_datetime(f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d} 09:00:00")
        dt_end = pd.to_datetime(f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d} 15:30:00")
        ax.set_xlim(dt_start, dt_end)

    ax.grid(axis="y")
    ax.legend(fontsize=7)

    plt.tight_layout()
    plt.show()


def plot_obs_trend(df: pd.DataFrame, n: int, list_ylabel: list):
    fig = plt.figure(figsize=(15, 12))
    ax = dict()
    gs = fig.add_gridspec(
        n, 1,
        wspace=0.0, hspace=0.0,
        height_ratios=[1 if i < n - 7 else 0.5 for i in range(n)]
    )
    for i, axis in enumerate(gs.subplots(sharex="col")):
        ax[i] = axis
        ax[i].set_xlim(0, 19500)
        ax[i].grid()

    for i in range(n):
        ax[i].plot(df[i])
        y_min, y_max = ax[i].get_ylim()
        if i < n - 7:
            if -1.1 < y_min:
                y_min = -1.1
            if y_max < 1.1:
                y_max = 1.1
            ax[i].set_ylim(y_min, y_max)
        else:
            if y_max < 1.1:
                y_max = 1.1
            ax[i].set_ylim(-0.1, y_max)
        ax[i].set_ylabel(list_ylabel[i])
    plt.tight_layout()
    plt.show()


def plot_reward_distribution(ser: pd.Series, logscale: bool = False):
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.hist(ser, bins=20)
    ax.set_title("Reward Distribution")
    ax.set_xlabel("Reward")

    if logscale:
        ax.set_yscale("log")
        ax.set_ylabel("Freq. in Log scale")
    else:
        ax.set_ylabel("Frequency")

    x_low, x_high = ax.get_xlim()
    if -1.0 < x_low:
        x_low = -1.0
    if x_high < 1.0:
        x_high = 1.0
    ax.set_xlim(x_low, x_high)
    ax.grid()

    plt.tight_layout()
    plt.show()


def plot_tick_trend(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(15, 3))

    ax.plot(df["Price"])
    ax.set_ylabel("Price")
    ax.grid()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    res = AppRes()
    agent = PPOAgentSB3()

    # 推論用データ
    # file = "ticks_20250819.xlsx"
    # file = "ticks_20250828.xlsx"
    file = "ticks_20251006.xlsx"
    # file = "ticks_20251009.xlsx"
    code = "7011"

    print(f"過去データ {file} の銘柄 {code} について推論します。")
    # Excel ファイルのフルパス
    path_excel = get_collection_path(res, file)
    # Excel ファイルをデータフレームに読み込む
    df = get_excel_sheet(path_excel, code)
    path_model = get_ppo_model_path(res, code)

    result = agent.infer(df, path_model)
    if not result:
        sys.exit("正常終了しませんでした。")

    # 取引結果
    df_transaction: pd.DataFrame = agent.results["transaction"]
    print(df_transaction)
    print(f"一株当りの損益 : {df_transaction['損益'].sum()} 円")
    plot_bar_profit(df_transaction)

    print("モデルへの報酬分布")
    ser_reward = pd.Series(agent.results["reward"])
    print(
        f"n: {len(ser_reward)}, "
        f"mean: {ser_reward.mean():.3f}, "
        f"stdev: {ser_reward.std():.3f}"
    )

    # 報酬分布
    plot_reward_distribution(ser_reward, logscale=True)

    # ティックトレンド
    plot_tick_trend(df)

    # 観測値トレンド
    df_obs = pd.concat([pd.Series(row) for row in agent.results["obs"]], axis=1).T
    rows = df_obs.shape[1]
    print(f"観測数 : {rows}")
    list_name = [
        "株価比",
        "MAΔ1",
        "MAΔ2",
        "RSI",
        "VWAPΔ",
        "含損益",
        "含損益M",
        "HOLD1",
        "HOLD2",
        "TRADE",
        "NONE",
        "LONG",
        "SHORT"
    ]
    plot_obs_trend(df_obs, rows, list_name)
