"""
Project Kabuto のバックテスト用 CLI アプリの起動プログラム (Kayaba)
"""
import datetime

from funcs.logs import setup_logging
from modules.kayaba import Kayaba



if __name__ == "__main__":
    main_logger = setup_logging()

    # DOE
    name_doe = "doe-000"

    # 銘柄コード
    code = "9984"

    # 開始日
    dt_start = datetime.datetime(2026, 3, 1)

    # バックテスト用クラス (Kayaba) のインスタンス生成を起動
    app = Kayaba(name_doe, code, dt_start)
    app.run()


