import datetime

import mplfinance as mpf
import yfinance as yf

from modules.psar_conventional import ParabolicSAR
from widgets.chart import MplChart


class TechnicalDrawer:
    def __init__(self, chart: MplChart):
        self.chart = chart

    def draw(self, code: str, name: str):
        symbol = f"{code}.T"
        ticker = yf.Ticker(symbol)
        df0 = ticker.history(period="3y", interval="1d")
        psar = ParabolicSAR()
        psar.calc(df0)

        dt_last = df0.index[len(df0) - 1]
        tdelta_1y = datetime.timedelta(days=180)
        df = df0[df0.index >= dt_last - tdelta_1y].copy()

        # チャートのクリア
        self.chart.clearAxes()

        mm05 = df0["Close"].rolling(5).median()
        mm25 = df0["Close"].rolling(25).median()
        mm75 = df0["Close"].rolling(75).median()

        apds = [
            mpf.make_addplot(mm05[df.index], width=0.75, label=" 5d moving median", ax=self.chart.ax[0]),
            mpf.make_addplot(mm25[df.index], width=0.75, label="25d moving median", ax=self.chart.ax[0]),
            mpf.make_addplot(mm75[df.index], width=0.75, label="75d moving median", ax=self.chart.ax[0]),
            mpf.make_addplot(
                df["Bear"],
                type="scatter",
                marker="o",
                markersize=5,
                color="blue",
                label="down trend",
                ax=self.chart.ax[0]
            ),
            mpf.make_addplot(
                df["Bull"],
                type="scatter",
                marker="o",
                markersize=5,
                color="red",
                label="up trend",
                ax=self.chart.ax[0]
            ),
        ]
        mpf.plot(
            df,
            type="candle",
            style="default",
            volume=self.chart.ax[1],
            datetime_format="%m-%d",
            addplot=apds,
            xrotation=0,
            ax=self.chart.ax[0]
        )
        self.chart.ax[0].set_title(f"{name} ({code})")
        self.chart.ax[0].legend(loc="best", fontsize=8)

        # チャートの再描画
        self.chart.refreshDraw()