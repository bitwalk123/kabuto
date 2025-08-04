import argparse
import sys

from PySide6.QtWidgets import QApplication, QWidget


class Example(QWidget):
    def __init__(self, config_path, parent=None):
        super().__init__(parent)

        # 1. argparseで解析された引数を利用してウィジェットを初期化
        print(f"Using config file: {config_path}")
        # ここでconfig_pathを読み込んで、アプリケーションの設定を行う
        # 例: self.load_settings(config_path)

        self.setWindowTitle("PySide6 App with argparse")
        self.setGeometry(300, 300, 300, 200)


def main():
    # 2. QApplicationの前に引数を解析
    parser = argparse.ArgumentParser(description='アプリケーションの起動')
    parser.add_argument('-xl', '--excel',
                        dest='path_excel',
                        type=str,
                        default='targets.xlsm',
                        help='使用するRSS用Excelファイル（デフォルト: targets.xlsm）')
    parser.add_argument('--debug',
                        action='store_true',
                        help='デバッグモードを有効にする')

    args = parser.parse_args()

    # 3. QApplicationをインスタンス化
    # QApplicationはsys.argvを処理できるので、引数を渡すのが一般的です。
    app = QApplication(sys.argv)

    # 5. その他の引数をグローバルな設定として利用
    if args.debug:
        print("Debug mode is enabled globally.")
        # QApplicationやその他のモジュールにデバッグ設定を適用
        # 例: QLoggingCategory.setFilterRules('*.debug=true')

    # 4. 解析した引数を Example クラスのインスタンス化時に渡す
    # この例では、config_pathを Example のコンストラクタに渡しています。
    win = Example(config_path=args.config_path)
    win.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
