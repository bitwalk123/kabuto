import sys

from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QStyle,
    QToolBar,
    QToolButton,
)


class PlotReview(QMainWindow):
    def __init__(self):
        super().__init__()
        toolbar = QToolBar()
        self.addToolBar(toolbar)

        but_open = QToolButton()
        but_open.setText('Open')
        but_open.setToolTip('Open file')
        but_open.setIcon(
            self.style().standardIcon(
                QStyle.StandardPixmap.SP_DirOpenIcon
            )
        )
        but_open.clicked.connect(self.on_open_clicked)
        toolbar.addWidget(but_open)

    def on_open_clicked(self):
        dlg = QFileDialog()
        dlg.setOption(QFileDialog.Option.DontUseNativeDialog)
        if dlg.exec():
            filename = dlg.selectedFiles()[0]
            self.gen_chart(filename)
        else:
            print("Canceled!")

    def gen_chart(self, filename: str):
        fig = Figure(figsize=(6, 6), dpi=100)
        canvas = FigureCanvas(fig)  # 描画に必要
        ax = fig.add_subplot(111)

        #ax.plot(df['close'])

        # 画面に表示（layout.addWidget）せずに保存だけ実行
        fig.savefig("temp.png")


def main():
    # QApplication は sys.argv を処理するので、そのまま引数を渡すのが一般的。
    app = QApplication(sys.argv)

    win = PlotReview()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
