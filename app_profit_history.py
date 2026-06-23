# Profit Simulator
import sys

from PySide6.QtWidgets import (
    QApplication,
)

from tools.profit_history import ProfitHistory


def main():
    app = QApplication(sys.argv)
    ex = ProfitHistory()
    ex.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
