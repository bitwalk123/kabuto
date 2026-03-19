import json
import os

from structs.res import AppRes


def read_contents_from_json(file_json) -> dict:
    with open(file_json) as f:
        dict_psar = json.load(f)
    return dict_psar


def save_contents_to_json(file_json: str, dict_psar: dict):
    with open(file_json, "w") as f:
        json.dump(dict_psar, f)


def save_setting(res: AppRes, code: str, dict_param: dict):
    """
    銘柄コード指定で設定用パラメータを保存
    :param res:
    :param code:
    :param dict_param:
    :return:
    """
    path_json_setting = os.path.join(res.dir_conf, f"{code}.json")
    with open(path_json_setting, "w") as f:
        json.dump(dict_param, f, indent=2)
