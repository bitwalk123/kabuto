import sys

from PySide6.QtWidgets import QApplication

from funcs.logs import setup_logging
from modules.prophet import Prophet


def main():
    # QApplication は sys.argv を処理するので、そのまま引数を渡すのが一般的。
    app = QApplication(sys.argv)

    win = Prophet()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    # ロギング設定を適用（ルートロガーを設定）
    main_logger = setup_logging()
    main()
