import importlib

from models.abstract import AlgoTradeBase


def get_model_instance(full_model_name:str):
    """モジュールをインポートしてクラスを辞書に追加する補助関数"""
    try:
        model = importlib.import_module(full_model_name)
        importlib.reload(model)
        for attr_name in dir(model):
            attr = getattr(model, attr_name)
            if (isinstance(attr, type) and
                    issubclass(attr, AlgoTradeBase) and
                    attr is not AlgoTradeBase):
                return attr()
    except Exception as e:
        print(f"Failed to load parser {full_model_name}: {e}")
        raise
