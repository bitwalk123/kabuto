from funcs.logs import setup_logging
from apostle.apostle_main import Apostle


def main():
    obj = Apostle()
    obj.run()


if __name__ == "__main__":
    # ロギング設定を適用（ルートロガーを設定）
    main_logger = setup_logging()
    main()
