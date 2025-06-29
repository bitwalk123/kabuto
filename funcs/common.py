import datetime
import glob
import os
import re


def get_name_15min_chart(code: str, dt: datetime.datetime) -> str:
    year = dt.year
    month = dt.month
    day = dt.day
    return f"{year:4d}/{month:02d}{day:02d}_15min_chart_{code}.png"


def get_name_15min_chart_now(code: str) -> str:
    dt_now = datetime.datetime.now()
    year = dt_now.year
    month = dt_now.month
    day = dt_now.day
    return f"{year:4d}/{month:02d}{day:02d}_15min_chart_{code}.png"


def get_date_str_from_collections(file_excel:str)->str:
    pattern = re.compile(r".+ticks_([0-9]{4})([0-9]{2})([0-9]{2})\.xlsx")
    m = pattern.match(file_excel)
    if m:
        date_str = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    else:
        date_str = "1970-01-01"
    return date_str


def get_sources_for_collection(dir_path:str) -> list:
    """
    シミュレーション対象のファイルリストを返す
    :return:
    """
    list_excel = glob.glob(os.path.join(dir_path, "ticks_*.xlsx"))
    return list_excel
