import argparse
import logging
import sys

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QSplashScreen

from funcs.logs import setup_logging
from modules.kabuto import Kabuto


def determine_debug_mode(args: argparse.Namespace) -> bool:
    """デバッグモードの判定"""
    if args.debug:
        return True
    elif sys.platform == "win32":
        return False
    else:
        return True  # Windows以外はデバッグ・モード


def gen_parser_for_app_cmdline_options() -> argparse.ArgumentParser:
    """
    アプリケーションをコンソールから起動した際の
    コマンドライン・オプションを処理するパーサーの生成
    """
    parser = argparse.ArgumentParser(description="アプリケーションの起動")
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="デバッグモードを有効にする"
    )
    return parser


def show_splash_screen(app: QApplication) -> QSplashScreen | None:
    """スプラッシュスクリーンの表示"""
    splash_pix = QPixmap("splash_image.png")
    if splash_pix.isNull():
        return None

    splash = QSplashScreen(splash_pix, Qt.WindowType.WindowStaysOnTopHint)
    splash.show()

    app.processEvents()
    return splash


def main() -> None:
    parser = gen_parser_for_app_cmdline_options()
    args = parser.parse_args()
    debug = determine_debug_mode(args)

    app = QApplication(sys.argv)
    splash = show_splash_screen(app)

    win = Kabuto(debug)
    win.show()

    if splash is not None:  # ← None チェックを追加
        splash.finish(win)
    # if splash:  # 最低 2 秒は表示する
    #    QTimer.singleShot(1000, lambda: splash.finish(win))

    sys.exit(app.exec())


if __name__ == "__main__":
    # ロギング設定を適用
    main_logger: logging.Logger = setup_logging()
    main_logger.info("アプリケーションを起動します")
    main()
