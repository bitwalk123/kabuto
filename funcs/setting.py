import json
import os

from structs.res import AppRes


def get_default_setting() -> dict:
    # デフォルトのパラメータ設定
    return {
        "PERIOD_MA_1": 60,
        "PERIOD_MA_2": 600,
        "PERIOD_MR": 30,
        "THRESHOLD_MR": 7.0,
        "LOSSCUT_1": -1.0e8,
    }


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

