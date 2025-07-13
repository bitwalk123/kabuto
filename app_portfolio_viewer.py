import sys

from PySide6.QtWidgets import QApplication

from broker.portfolio_viewer import PortfolioViewer


def main():
    app = QApplication(sys.argv)
    win = PortfolioViewer()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
