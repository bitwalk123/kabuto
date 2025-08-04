import sys

from PySide6.QtWidgets import QApplication

from funcs.logs import setup_logging
from rhino.rhino_funcs import gen_parser_for_cmdline
from rhino.rhino_main import Rhino


def main():
    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
    # コンソールから起動した際のコマンドライン・オプション
    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
    # パーサーを作成
    parser = gen_parser_for_cmdline()
    # 引数をパース
    args = parser.parse_args()
    # デバッグ・モードの判定
    if args.debug:
        debug = True
    elif sys.platform == "win32":
        debug = False
    else:
        debug = True  # Windows以外はデバッグ・モード

    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
    # QApplicationをインスタンス化
    # _/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_
    # QApplication は sys.argv を処理するので、そのまま引数を渡すのが一般的。
    app = QApplication(sys.argv)

    win = Rhino(args.excel_path, debug)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    # ロギング設定を適用（ルートロガーを設定）
    main_logger = setup_logging()
    main()
