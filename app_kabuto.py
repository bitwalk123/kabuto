"""
Project Kabuto のデイトレ用 GUI アプリの起動プログラム (Kabuto)
"""
import argparse
import logging
import sys
import time

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


def show_splash_screen_with_timer(
        app: QApplication,
        win: Kabuto,
        min_duration: int = 1000
) -> None:
    """
    スプラッシュスクリーンを最低時間表示してからメインウィンドウを表示

    Args:
        app: QApplicationインスタンス
        win: メインウィンドウ
        min_duration: 最低表示時間（ミリ秒）
    """
    splash_pix = QPixmap("splash_image.png")
    if splash_pix.isNull():
        win.show()
        return

    splash = QSplashScreen(splash_pix, Qt.WindowType.WindowStaysOnTopHint)
    splash.show()
    app.processEvents()

    # タイマーで指定時間後にメインウィンドウを表示
    def show_main_window():
        splash.finish(win)
        win.show()

    QTimer.singleShot(min_duration, show_main_window)


def main() -> None:
    parser = gen_parser_for_app_cmdline_options()
    args = parser.parse_args()
    debug = determine_debug_mode(args)

    app = QApplication(sys.argv)
    win = Kabuto(debug)

    # スプラッシュを2秒間表示してからメインウィンドウ表示
    show_splash_screen_with_timer(app, win, min_duration=2000)

    sys.exit(app.exec())


if __name__ == "__main__":
    # ロギング設定を適用
    main_logger: logging.Logger = setup_logging()
    main_logger.info("アプリケーションを起動します")
    main()
