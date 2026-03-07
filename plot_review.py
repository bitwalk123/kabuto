import sys

from PySide6.QtWidgets import QApplication, QMainWindow, QToolBar, QStyle, QToolButton


class PlotReview(QMainWindow):
    def __init__(self):
        super().__init__()
        toolbar = QToolBar()
        self.addToolBar(toolbar)

        but_open = QToolButton()
        but_open.setText('Open')
        but_open.setToolTip('Open file')
        but_open.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        but_open.clicked.connect(self.on_open_clicked)
        toolbar.addWidget(but_open)

    def on_open_clicked(self):
        print("DEBUG!")


def main():
    # QApplication は sys.argv を処理するので、そのまま引数を渡すのが一般的。
    app = QApplication(sys.argv)

    win = PlotReview()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
