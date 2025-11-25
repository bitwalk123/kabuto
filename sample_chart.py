import sys

from PySide6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy, QApplication
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class ChartWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Matplotlib Figure と Canvas
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        # サイズポリシーを設定（拡張可能にする）
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        self.canvas.updateGeometry()

        # レイアウトに追加
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)
        self.setLayout(layout)


def main():
    app = QApplication(sys.argv)
    win = ChartWidget()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
