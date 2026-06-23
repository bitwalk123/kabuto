# Profit Simulator
import sys

from PySide6.QtWidgets import (
    QApplication,
)

from tools.profit_sim_app import ProfitSimulatorApp


def main():
    app = QApplication(sys.argv)
    ex = ProfitSimulatorApp()
    ex.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()