import argparse
import sys

from PySide6.QtWidgets import QApplication

from funcs.logs import setup_logging
from rhino.rhino_main import Rhino


def main():
    # コンソールから起動した際のコマンドライン・オプション
    parser = argparse.ArgumentParser(description="アプリケーションの起動")
    # 使用するRSS用Excelファイル（デフォルト: targets.xlsm）
    parser.add_argument(
        "-xl", "--excel",
        dest="excel_path",
        type=str,
        default="targets.xlsm",
        help="使用するRSS用Excelファイル（デフォルト: targets.xlsm）"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="デバッグモードを有効にする"
    )
    args = parser.parse_args()

    if args.debug:
        debug = True
    elif sys.platform == "win32":
        debug = False
    else:
        debug = True  # Windows以外はデバッグ・モード

    app = QApplication(sys.argv)
    win = Rhino(args.excel_path, debug)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    # ロギング設定を適用（ルートロガーを設定）
    main_logger = setup_logging()
    main()
