import sys

import pyqtgraph as pg
from PySide6.QtWidgets import QApplication, QMainWindow, QToolBar, QToolButton, QStyle


class SampleChart(pg.PlotWidget):
    def __init__(self):
        super().__init__()

class SampleTaLib(QMainWindow):
    def __init__(self):
        super().__init__()

        toolbar=QToolBar()
        but_play = QToolButton()
        but_play.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        but_play.clicked.connect(self.on_play_clicked)
        toolbar.addWidget(but_play)
        self.addToolBar(toolbar)

        self.chart = chart = SampleChart()
        self.setCentralWidget(chart)

    def on_play_clicked(self):
        pass

def main():
    app = QApplication(sys.argv)
    win = SampleTaLib()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
