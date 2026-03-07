import sys

from PySide6.QtWidgets import QApplication, QMainWindow, QToolBar


class PlotReview(QMainWindow):
    def __init__(self):
        super().__init__()
        toolbar = QToolBar()
        self.addToolBar(toolbar)





def main():
    # QApplication は sys.argv を処理するので、そのまま引数を渡すのが一般的。
    app = QApplication(sys.argv)

    win = PlotReview()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
