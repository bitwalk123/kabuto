import importlib
import logging
from typing import Any

from models.abstract import AlgoTradeBase
from structs.res import AppRes

logger = logging.getLogger(__name__)
res = AppRes()


def get_model_instance(name_model: str, dict_model: dict[str, Any]):
    """モジュールをインポートしてクラスを辞書に追加"""
    full_model_name = ".".join([res.dir_model, name_model])
    try:
        model = importlib.import_module(full_model_name)
        importlib.reload(model)
        for attr_name in dir(model):
            attr = getattr(model, attr_name)
            if (isinstance(attr, type) and
                    issubclass(attr, AlgoTradeBase) and
                    attr is not AlgoTradeBase):
                if name_model == getattr(attr, "MODEL_NAME"):
                    dict_model[name_model] = attr()
                    logger.info(f"モデル '{name_model}' を読み込みました。")
                else:
                    logger.error(
                        f"Model '{name_model}' is not matched "
                        f"with {getattr(attr, "MODEL_NAME")}!"
                    )
                    raise
    except Exception as e:
        logger.error(f"Failed to load model {full_model_name}: {e}")
        raise
