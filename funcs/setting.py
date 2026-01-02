import json
import os

from structs.res import AppRes


def get_default_setting() -> dict:
    # デフォルトのパラメータ設定
    return {
        "PERIOD_MA_1": 60,
        "PERIOD_MA_2": 600,
        "PERIOD_SLOPE": 5,
        "THRESHOLD_SLOPE": 0.5,  # doe-9a
        "LOSSCUT_1": -25,
    }


def get_trend_footer(dict_ts: dict, dict_setting: dict) -> str:
    return (
        f"DATE = {dict_ts['datetime_str_2']} / "
        f"PERIOD_MA_1 = {dict_setting['PERIOD_MA_1']} / "
        f"PERIOD_MA_2 = {dict_setting['PERIOD_MA_2']} / "
        f"PERIOD_SLOPE = {dict_setting['PERIOD_SLOPE']} / "
        f"THRESHOLD_SLOPE = {dict_setting['THRESHOLD_SLOPE']} / "
        f"LOSSCUT_1 = {dict_setting['LOSSCUT_1']}"
    )


def load_setting(res: AppRes, code: str) -> dict:
    """
    銘柄コード指定で設定用 JSON ファイルのロード
    :param res:
    :param code:
    :return:
    """
    path_json_setting = os.path.join(res.dir_conf, f"{code}.json")
    if os.path.exists(path_json_setting):
        with open(path_json_setting) as f:
            dict_setting = json.load(f)
        return dict_setting
    else:
        return get_default_setting()
