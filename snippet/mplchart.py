import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import mplfinance as mpf
import pandas as pd
import yfinance as yf


def chart_daily(df: pd.DataFrame) -> None:
    plt.rcParams["font.family"] = "monospace"
    plt.rcParams["font.size"] = 8

    fig = Figure(figsize=(8, 3), dpi=100)
    canvas = FigureCanvas(fig)
    ax = dict()
    n = 2
    gs = fig.add_gridspec(
        n, 1, wspace=0.0, hspace=0.0, height_ratios=[2 if i == 0 else 1 for i in range(n)]
    )
    for i, axis in enumerate(gs.subplots(sharex="col")):
        ax[i] = axis
        ax[i].grid()

    mpf.plot(
        df,
        type="candle",
        style="default",
        datetime_format="%m/%d",
        xrotation=0,
        update_width_config=dict(candle_linewidth=0.75),
        volume=ax[1],
        ax=ax[0],
    )

    if "shortName" in yticker.info:
        ax[0].set_title(f"{yticker.info['shortName']} ({symbol})")
    elif "longName" in yticker.info:
        ax[0].set_title(f"{yticker.info['longName']} ({symbol})")
    else:
        ax[0].set_title(f"{symbol}")

    canvas.draw()
    fig.savefig("test.png")
    plt.show()


if __name__ == "__main__":
    code = "N225"
    symbol = f"^{code}"
    yticker = yf.Ticker(symbol)
    df = yticker.history(period="3mo", interval="1d")
    chart_daily(df)
