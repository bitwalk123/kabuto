from funcs.logs import setup_logging
from modules.lagrange import Lagrange


def main():
    obj = Lagrange()
    obj.run()


if __name__ == "__main__":
    # ロギング設定を適用（ルートロガーを設定）
    main_logger = setup_logging()
    main()
