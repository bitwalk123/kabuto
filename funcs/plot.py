import pandas as pd
from matplotlib import font_manager as fm, pyplot as plt, dates as mdates


def plot_mpl_chart(df: pd.DataFrame, title: str, condition: str, imgname: str):
    FONT_PATH = "fonts/RictyDiminished-Regular.ttf"
    fm.fontManager.addfont(FONT_PATH)

    # FontPropertiesオブジェクト生成（名前の取得のため）
    font_prop = fm.FontProperties(fname=FONT_PATH)
    font_prop.get_name()

    plt.rcParams["font.family"] = font_prop.get_name()
    plt.rcParams["font.size"] = 16
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.grid()

    ax.plot(df["Price"], color="gray", linewidth=0.5)

    # Parabolic SAR
    ser_bull = df[df["Trend"] > 0]["PSAR"]
    ser_bear = df[df["Trend"] < 0]["PSAR"]
    ax.scatter(x=ser_bull.index, y=ser_bull, s=5, c="red")
    ax.scatter(x=ser_bear.index, y=ser_bear, s=5, c="blue")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_title(condition, fontsize="small")

    plt.suptitle(title)
    #plt.tight_layout()
    plt.savefig(imgname)
    plt.close()
