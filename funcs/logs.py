import logging
import logging.handlers
import os


def setup_logging():
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # ルートロガーの取得と設定
    # アプリケーション全体で使うロガー。名前を指定しないとルートロガーになる。
    logger = logging.getLogger()  # ここが変更点！
    logger.setLevel(logging.DEBUG)  # 全てのレベルを出力するためDEBUGに設定

    # 既存のハンドラがあれば削除（二重出力防止のため）
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # コンソールハンドラ
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)  # コンソールにはINFO以上
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(console_formatter)
    logger.addHandler(ch)

    # ファイルハンドラ
    log_file_path = os.path.join(log_dir, "app.log")
    fh = logging.handlers.RotatingFileHandler(
        log_file_path,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    fh.setLevel(logging.DEBUG)  # ファイルにはDEBUGレベルも出力
    # ロガー名も出力するフォーマットに変更
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(lineno)d - %(message)s')
    fh.setFormatter(file_formatter)
    logger.addHandler(fh)

    # xlwingsのログも設定
    xlwings_logger = logging.getLogger("xlwings")
    for handler in xlwings_logger.handlers[:]:
        xlwings_logger.removeHandler(handler)
    xlwings_logger.setLevel(logging.INFO)
    xlwings_logger.addHandler(fh)  # 同じファイルハンドラを使う

    return logger
