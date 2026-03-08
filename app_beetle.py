import logging
import sys

from PySide6.QtWidgets import QApplication

from funcs.logs import setup_logging
from modules.beetle import Beetle


def main():
    # QApplication は sys.argv を処理するので、そのまま引数を渡すのが一般的。
    app = QApplication(sys.argv)

    win = Beetle()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    # ロギング設定を適用
    main_logger: logging.Logger = setup_logging()
    main_logger.info("アプリケーションを起動します")
    main()
