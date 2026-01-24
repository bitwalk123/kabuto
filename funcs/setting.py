import json
import logging
import os
import warnings
from pathlib import Path

import requests
from urllib3.exceptions import InsecureRequestWarning

from structs.res import AppRes

logger = logging.getLogger(__name__)

# HTTPS verify=False の警告を抑制
warnings.simplefilter("ignore", InsecureRequestWarning)


def get_default_setting() -> dict:
    # デフォルトのパラメータ設定
    return {
        "PERIOD_WARMUP": 60,
        "PERIOD_MA_1": 60,
        "PERIOD_MA_2": 600,
        "PERIOD_RR": 30,
        "THRESHOLD_SLOPE": 1.0,  # doe-10
        "PERIOD_SLOPE": 5,
        "TURBULENCE": 20,
        "LOSSCUT_1": -25,
        "THRESHOLD_PM_MIN": 17.5,
        "THRESHOLD_DDR_MIN": 0.4,
        "N_MINUS_MAX": 180,
    }


def get_trend_footer(dict_ts: dict, dict_setting: dict) -> str:
    return (
        f"DATE = {dict_ts['datetime_str_2']} / "
        f"PERIOD_WARMUP = {dict_setting['PERIOD_WARMUP']} / "
        f"PERIOD_MA_1 = {dict_setting['PERIOD_MA_1']} / "
        f"PERIOD_MA_2 = {dict_setting['PERIOD_MA_2']} / "
        f"PERIOD_SLOPE = {dict_setting['PERIOD_SLOPE']} / "
        f"THRESHOLD_SLOPE = {dict_setting['THRESHOLD_SLOPE']} / "
        f"PERIOD_RR = {dict_setting['PERIOD_RR']} / "
        f"TURBULENCE = {dict_setting['TURBULENCE']} / "
        f"LOSSCUT_1 = {dict_setting['LOSSCUT_1']} / "
        f"THRESHOLD_PM_MIN = {dict_setting['THRESHOLD_PM_MIN']} / "
        f"THRESHOLD_DDR_MIN = {dict_setting['THRESHOLD_DDR_MIN']} /"
        f"N_MINUS_MAX = {dict_setting['N_MINUS_MAX']}"
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


def update_setting(res: AppRes, code: str):
    """
    最新の JSON ファイルを HTTP サーバーからダウンロードして更新
    :param res:
    :param code:
    :return:
    """
    url = f"{res.url_conf}/{code}.json"

    local_conf_dir = Path(res.dir_conf)
    local_conf_dir.mkdir(exist_ok=True)

    try:
        r = requests.get(url, timeout=3, verify=False)
        if r.status_code == 200:
            data = r.json()
            with open(local_conf_dir / f"{code}.json", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"{__name__}: {code}.json を更新しました")
        else:
            logger.warning(f"{__name__}: リモートに {code}.json はありません (status={r.status_code})")
    except Exception as e:
        logger.error(f"{__name__}: {code}.json の更新に失敗しました: {e}")
