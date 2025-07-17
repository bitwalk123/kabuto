from funcs.logs import setup_logging
from lagrange.lagrange_main import Lagrange


def main():
    obj = Lagrange()
    obj.run()


if __name__ == "__main__":
    # ロギング設定を適用（ルートロガーを設定）
    main_logger = setup_logging()
    main()
