# MarketSPEED 2 RSS 用いた信用取引テスト
import sys

from PySide6.QtWidgets import QApplication

from funcs.logs import setup_logging
from matisse.matisse_main import Matisse


def main():
    app = QApplication(sys.argv)
    win = Matisse()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    # ロギング設定を適用（ルートロガーを設定）
    main_logger = setup_logging()
    main()
