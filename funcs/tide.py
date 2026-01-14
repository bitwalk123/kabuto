import datetime
import os
import re


def conv_datetime_from_timestamp(ts) -> str:
    dt = datetime.datetime.fromtimestamp(ts)
    return f"{dt.year:04}-{dt.month:02}-{dt.day:02} {dt.hour:02}:{dt.minute:02}:{dt.second:02}"


def get_date_str_from_file(file: str) -> str:
    """
    ファイル名から日付文字列を返す関数
    :param file:
    :return:
    """
    pattern = re.compile(r".+_(\d{8})\..+")
    if m := pattern.match(file):
        return m.group(1)
    else:
        return "19700101"


def get_year_date_str_from_file(file: str) -> str:
    """
    ファイル名から日付文字列を返す関数
    :param file:
    :return:
    """
    pattern = re.compile(r".+_(\d{4})(\d{4})\..+")
    if m := pattern.match(file):
        return os.path.join(m.group(1), m.group(2))
    else:
        return os.path.join("1970", "0101")


def get_date_dir_from_file(file: str) -> str:
    """
    ファイル名から年/月/日の階層ディレクトリ文字列を返す関数
    :param file:
    :return:
    """
    pattern = re.compile(r".+_(\d{4})(\d{2})(\d{2})\..+")
    if m := pattern.match(file):
        return os.path.join(m.group(1), m.group(2), m.group(3))
    else:
        return os.path.join("1970", "01", "01")


def get_date_str_today() -> str:
    dt = datetime.datetime.now()
    return f"{dt.year:04}{dt.month:02}{dt.day:02}"


def get_datetime_str() -> str:
    global datetime_str
    dt = datetime.datetime.now()
    return f"{dt.year:04d}{dt.month:02d}{dt.day:02d}{dt.hour:02d}{dt.minute:02d}{dt.second:02d}"


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
        pattern = re.compile(r".+_([0-9]{4})([0-9]{2})([0-9]{2}).*\.xlsx")
        m = pattern.match(excel_path)
        if m:
            year = int(m.group(1))
            month = int(m.group(2))
            day = int(m.group(3))
        else:
            year = 1970
            month = 1
            day = 1
            print("正規表現にマッチしませんでした！")

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
    dict_ts["datetime_str"] = f"{year:04}{month:02}{day:02}"
    dict_ts["datetime_str_2"] = f"{year:04}-{month:02}-{day:02}"

    return dict_ts
