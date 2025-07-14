import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton, MouseEvent
from matplotlib.backends.backend_qtagg import (
    NavigationToolbar2QT as NavigationToolbar,
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector

from PySide6.QtCore import QObject, Signal

FONT_PATH = 'fonts/RictyDiminished-Regular.ttf'


class CandleChartSignal(QObject):
    rectangleSelected = Signal(list)


class CandleChart(FigureCanvas):
    def __init__(self):
        self.fig = Figure()
        super().__init__(self.fig)

        # Constants
        self.signal = CandleChartSignal()
        self.rs: RectangleSelector | None = None

        # Font setting
        fm.fontManager.addfont(FONT_PATH)
        font_prop = fm.FontProperties(fname=FONT_PATH)
        plt.rcParams['font.family'] = font_prop.get_name()
        plt.rcParams['font.size'] = 14

        # Plot margin
        self.fig.subplots_adjust(
            left=0.075,
            right=0.90,
            top=0.95,
            bottom=0.05,
        )

        # Axes
        self.ax = dict()

        # Selector
        # self.init_rectangle_selector(ax)

    def clearAxes(self):
        axs = self.fig.axes
        for ax in axs:
            ax.cla()

    def clearRectangle(self):
        self.rs.clear()

    def initAxes(self, ax, n: int):
        if n > 1:
            gs = self.fig.add_gridspec(
                n, 1,
                wspace=0.0, hspace=0.0,
                height_ratios=[3 if i == 0 else 1 for i in range(n)]
            )
            for i, axis in enumerate(gs.subplots(sharex='col')):
                ax[i] = axis
                ax[i].grid()
        else:
            ax[0] = self.fig.add_subplot()
            ax[0].grid()

        # Selector
        self.init_rectangle_selector(ax[0])

    def initChart(self, n: int):
        self.removeAxes()
        self.initAxes(self.ax, n)

    def init_rectangle_selector(self, ax):
        self.rs = RectangleSelector(
            ax,
            self.selection,
            useblit=True,
            button=MouseButton.LEFT,  # disable middle & right buttons
            minspanx=5,
            minspany=5,
            spancoords='pixels',
            interactive=True,
            props=dict(
                facecolor='#eef',
                edgecolor='blue',
                alpha=0.2,
                fill=True,
            )
        )

    def refreshDraw(self):
        self.fig.canvas.draw()

    def removeAxes(self):
        axs = self.fig.axes
        for ax in axs:
            ax.remove()

    def selection(self, eclick: MouseEvent, erelease: MouseEvent):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        self.signal.rectangleSelected.emit([x1, y1, x2, y2])


class ChartNavigation(NavigationToolbar):
    def __init__(self, chart: FigureCanvas):
        super().__init__(chart)