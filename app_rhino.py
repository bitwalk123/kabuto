import argparse
import sys

from PySide6.QtWidgets import QApplication

from funcs.logs import setup_logging
from rhino.rhino_main import Rhino


def main():
    parser = argparse.ArgumentParser(description='アプリケーションの起動')
    parser.add_argument(
        '--debug',
        action='store_true',
        help='デバッグモードを有効にする'
    )
    args = parser.parse_args()

    if args.debug:
        debug = True
    elif sys.platform == "win32":
        debug = False
    else:
        debug = True  # Windows 以外はデバッグ・モード

    app = QApplication(sys.argv)
    win = Rhino(debug)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    # ロギング設定を適用（ルートロガーを設定）
    main_logger = setup_logging()
    main()
