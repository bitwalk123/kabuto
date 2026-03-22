"""
Project Kabuto のバックテスト用 CLI アプリの起動プログラム (Kayaba)
"""
import datetime
import logging
import time

from funcs.logs import setup_logging
from modules.kayaba import Kayaba

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    main_logger = setup_logging()

    # DOE
    name_doe = "doe-000"

    # 銘柄コード
    code = "9984"

    # 開始日
    dt_start = datetime.datetime(2026, 3, 19)

    # バックテスト用クラス (Kayaba) のインスタンス生成を起動
    start = time.perf_counter()
    app = Kayaba(name_doe, code, dt_start)
    app.run()
    duration = time.perf_counter() - start

    logger.info(f"バックテストを終了しました。［{duration / 60:.2f} 分］")



