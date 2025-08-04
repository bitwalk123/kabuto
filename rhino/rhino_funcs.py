import argparse
import datetime
import re


def gen_parser_for_cmdline() -> argparse.ArgumentParser:
    """
    コンソールから起動した際のコマンドライン・オプションを処理するパーサーの生成
    :return:
    """
    # パーサーを作成
    parser = argparse.ArgumentParser(description="アプリケーションの起動")

    # 使用するRSS用Excelファイル（デフォルト: targets.xlsm）
    # RSS = Realtime Spread Sheet
    parser.add_argument(
        "-xl", "--excel",
        dest="excel_path",
        type=str,
        default="targets.xlsm",
        help="使用するRSS用Excelファイル（デフォルト: targets.xlsm）"
    )

    # デバッグモード用フラグ
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="デバッグモードを有効にする"
    )

    return parser


def get_intraday_timestamp(excel_path: str = "") -> dict:
    """
    ザラ場の開始時間などのタイムスタンプを算出
    :param excel_path:
    :return:
    """
    if excel_path == "":
        dt = datetime.datetime.now()
        year = dt.year
        month = dt.month
        day = dt.day
    else:
        # _________________________________________________________________
        # ［デバッグ用］Excel のファイル名より日付情報を抽出
        pattern = re.compile(r".*tick_([0-9]{4})([0-9]{2})([0-9]{2}).*\.xlsx")
        m = pattern.match(excel_path)
        if m:
            year = int(m.group(1))
            month = int(m.group(2))
            day = int(m.group(3))
        else:
            year = 1970
            month = 1
            day = 1

    dt_start = datetime.datetime(year, month, day, hour=9, minute=0)
    dt_end_1h = datetime.datetime(year, month, day, hour=11, minute=30)
    dt_start_2h = datetime.datetime(year, month, day, hour=12, minute=30)
    dt_end_2h = datetime.datetime(year, month, day, hour=15, minute=24, second=50)
    dt_ca = datetime.datetime(year, month, day, hour=15, minute=25)
    dt_end = datetime.datetime(year, month, day, hour=15, minute=30)

    dict_ts = dict()
    # タイムスタンプに変換してインスタンス変数で保持
    dict_ts["start"] = dt_start.timestamp()
    dict_ts["end_1h"] = dt_end_1h.timestamp()
    dict_ts["start_2h"] = dt_start_2h.timestamp()
    dict_ts["end_2h"] = dt_end_2h.timestamp()
    dict_ts["ca"] = dt_ca.timestamp()
    dict_ts["end"] = dt_end.timestamp()
    # 日付文字列
    dict_ts["date_str"] = f"{year:04}{month:02}{day:02}"

    return dict_ts
