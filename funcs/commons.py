import datetime
import glob
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

from structs.res import AppRes


def check_doe_factor_match(df_doe: pd.DataFrame, dict_setting: dict[str, Any]):
    """
    df_doe.columns と dict_setting のキーが完全に一致（過不足なし）しているか確認する。
    順序は問わない。
    """
    return set(df_doe.columns) == set(dict_setting.keys())


def detect_cross(prev: float | None, curr: float) -> float:
    """
    移動平均の乖離の符号変化からクロスを検出
    :param prev:
    :param curr:
    :return:
    """
    if prev is None:
        return 0.0
    if prev < 0 < curr:
        return +1.0
    if curr < 0 < prev:
        return -1.0
    return 0.0


def get_collection_path(res: AppRes, file: str) -> str:
    path_excel = str(Path(os.path.join(res.dir_collection, file)).resolve())
    return path_excel


def get_name_15min_chart(code: str, dt: datetime.datetime) -> str:
    year = f"{dt.year:04d}"
    month = f"{dt.month:02d}"
    day = f"{dt.day:02d}"
    output_dir = os.path.join(year, month, day)
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"15min_chart_{code}.png")


def get_name_15min_chart_now(code: str) -> str:
    dt_now = datetime.datetime.now()
    year = dt_now.year
    month = dt_now.month
    day = dt_now.day
    return f"{year:4d}/{month:02d}{day:02d}_15min_chart_{code}.png"


def get_name_15min_chart_usd(code: str, dt: datetime.datetime) -> str:
    year = f"{dt.year:04d}"
    month = f"{dt.month:02d}"
    day = f"{dt.day:02d}"
    output_dir = os.path.join(year, month, day)
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"15min_chart_{code}_usd.png")


def get_date_str_from_collections(file_excel: str) -> str:
    pattern = re.compile(r".+ticks_([0-9]{4})([0-9]{2})([0-9]{2})\.xlsx")
    m = pattern.match(file_excel)
    if m:
        date_str = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    else:
        date_str = "1970-01-01"
    return date_str


def get_datestr_from_collections(file_excel: str) -> str:
    pattern = re.compile(r".+ticks_([0-9]{4})([0-9]{2})([0-9]{2})\.xlsx")
    m = pattern.match(file_excel)
    if m:
        date_str = f"{m.group(1)}{m.group(2)}{m.group(3)}"
    else:
        date_str = "19700101"
    return date_str


def get_dt_from_collections(file_excel: str) -> datetime.datetime:
    pattern = re.compile(r".+ticks_([0-9]{4})([0-9]{2})([0-9]{2})\.xlsx")
    m = pattern.match(file_excel)
    if m:
        dt = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    else:
        dt = datetime.datetime(1970, 1, 1)
    return dt


def get_date_str_from_report(file_excel: str) -> str:
    pattern = re.compile(r".+report_([0-9]{4}-[0-9]{2}-[0-9]{2})\.csv")
    m = pattern.match(file_excel)
    if m:
        date_str = m.group(1)
    else:
        date_str = "1970-01-01"
    return date_str


def get_sources_for_collection(dir_path: str) -> list[str]:
    """
    シミュレーション対象のファイルリストを返す
    :return:
    """
    list_excel = glob.glob(os.path.join(dir_path, "ticks_*.xlsx"))
    return sorted(list_excel)


def init_transaction() -> dict[str, list]:
    """
    取引明細用データ辞書の初期化
    :return:
    """
    return {
        "注文日時": [],
        "銘柄コード": [],
        "売買": [],
        "約定単価": [],
        "約定数量": [],
        "損益": [],
    }
