import json
import logging
import os
import warnings
from pathlib import Path

import requests
from urllib3.exceptions import InsecureRequestWarning

from funcs.ios import save_setting
from structs.res import AppRes

logger = logging.getLogger(__name__)

# HTTPS verify=False の警告を抑制
warnings.simplefilter("ignore", InsecureRequestWarning)


def get_default_setting() -> dict:
    # デフォルトのパラメータ設定
    return {
        "PERIOD_WARMUP": 60,  # 寄り付き後のウォームアップ期間
        "PERIOD_MA_1": 30,  # 短周期移動平均線の周期
        "PERIOD_MA_2": 300,  # 長周期移動平均線の周期
        "LOSSCUT_1": -25.0,  # 単純ロスカットをするためのしきい値
        "N_MINUS_MAX": 90,  # 含み損益が連続マイナスを許容する最大回数
        "DD_PROFIT": 5.0,  # 「含み益最大値」がこれを超えればドローダウン対象
        "DD_RATIO": 0.5,  # ドローダウン比率がこのしきい値を超えれば利確
    }


def get_trend_footer(dict_ts: dict, dict_setting: dict) -> str:
    return (
        f"DATE = {dict_ts['datetime_str_2']} / "
        f"PERIOD_WARMUP = {dict_setting['PERIOD_WARMUP']} / "
        f"PERIOD_MA_1 = {dict_setting['PERIOD_MA_1']} / "
        f"PERIOD_MA_2 = {dict_setting['PERIOD_MA_2']} / "
        f"LOSSCUT_1 = {dict_setting['LOSSCUT_1']} / "
        f"N_MINUS_MAX = {dict_setting['N_MINUS_MAX']} / "
        f"DD_PROFIT = {dict_setting['DD_PROFIT']} / "
        f"DD_RATIO = {dict_setting['DD_RATIO']}"
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
            save_setting(res, code, get_default_setting())
            logger.info(f"{__name__}: デフォルトの設定で {code}.json を保存しました。")

    except Exception as e:
        logger.error(f"{__name__}: {code}.json の更新に失敗しました: {e}")
