# Profit Simulator
import sys

from PySide6.QtWidgets import (
    QApplication,
)

from tools.profit_sim import ProfitSimulator


def main():
    app = QApplication(sys.argv)
    ex = ProfitSimulator()
    ex.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()