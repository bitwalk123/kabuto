"""
Project Kabuto のバックテスト用 CLI アプリの起動プログラム (Kayaba)
"""
import argparse
import datetime
import logging
import time

from funcs.logs import setup_logging
from modules.kayaba import Kayaba

logger = logging.getLogger(__name__)


def main():
    # 1. パーサーの作成
    parser = argparse.ArgumentParser(description="Kayaba の起動オプション")
    # 2. オプションの追加 (-a が指定されたら args.a を True にする)
    parser.add_argument('-a', '--all', action='store_true', help="All オプションを有効にします")
    # 3. 引数の解析
    args = parser.parse_args()

    # DOE
    name_doe = "doe-001"

    # 開始日
    if args.a:
        dt_start = datetime.datetime(2026, 2, 1)
    else:
        dt_now = datetime.datetime.now()
        dt_start = datetime.datetime(dt_now.year, dt_now.month, dt_now.day)

    # データスコープ
    logger.info(f"{dt_start} 以降のティックデータを対象にします。")

    # 銘柄コード
    for code in ["9984"]:
        start = time.perf_counter()
        # バックテスト用クラス (Kayaba) のインスタンスを生成して起動
        app = Kayaba(name_doe, code, dt_start)
        app.run()
        duration = time.perf_counter() - start
        logger.info(f"{code} のバックテストが終了しました。［{duration / 60:.2f} 分］")


if __name__ == "__main__":
    main_logger = setup_logging()
    main()
