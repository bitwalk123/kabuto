import sys
from PySide6.QtWidgets import QApplication

from funcs.logs import setup_logging
from broker.server import StockBroker


def main():
    app = QApplication(sys.argv)
    win = StockBroker()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    # ロギング設定を適用（ルートロガーを設定）
    main_logger = setup_logging()
    main()
