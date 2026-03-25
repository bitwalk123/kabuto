import importlib
import logging
from typing import Dict

from models.abstract import AlgoTradeBase
from structs.res import AppRes

logger = logging.getLogger(__name__)
res = AppRes(False)


def get_model_instance(name_model: str) -> AlgoTradeBase:
    """
    モデルモジュールをインポートしてインスタンスを返す

    Args:
        name_model: モデル名（ファイル名から.pyを除いたもの）

    Returns:
        AlgoTradeBase: モデルインスタンス

    Raises:
        ImportError: モジュールのインポートに失敗
        AttributeError: MODEL_NAME属性が存在しない
        ValueError: MODEL_NAMEが一致しない、または有効なクラスが見つからない

    Example:
        >>> model = get_model_instance("algo_trade")
        >>> action, info = model.predict(obs, masks)
    """
    full_model_name = f"{res.dir_model}.{name_model}"

    try:
        # モジュールをインポート（既にインポート済みなら再読み込み）
        model_module = importlib.import_module(full_model_name)
        importlib.reload(model_module)

        # モジュール内の全属性を走査
        for attr_name in dir(model_module):
            attr = getattr(model_module, attr_name)

            # AlgoTradeBaseを継承したクラスか確認
            if not (isinstance(attr, type) and
                    issubclass(attr, AlgoTradeBase) and
                    attr is not AlgoTradeBase):
                continue

            # MODEL_NAME属性の確認
            model_name_attr = getattr(attr, "MODEL_NAME", None)

            if model_name_attr is None:
                logger.warning(
                    f"Class '{attr_name}' in '{name_model}' "
                    f"has no MODEL_NAME attribute, skipping"
                )
                continue

            # MODEL_NAMEがファイル名と一致するか確認
            if name_model == model_name_attr:
                logger.info(f"モデル '{name_model}' を読み込みました。")
                return attr()  # インスタンスを返す
            else:
                logger.warning(
                    f"MODEL_NAME mismatch in '{attr_name}': "
                    f"expected '{name_model}', got '{model_name_attr}', skipping"
                )

        # ループが終わってもモデルが見つからなかった
        raise ValueError(
            f"No valid AlgoTradeBase subclass with MODEL_NAME='{name_model}' "
            f"found in module '{full_model_name}'"
        )

    except ImportError as e:
        logger.error(f"Failed to import module '{full_model_name}': {e}")
        raise
    except (AttributeError, ValueError) as e:
        logger.error(f"Error loading model '{name_model}': {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading model '{name_model}': {e}")
        raise


def load_model_into_dict(
        name_model: str,
        dict_model: Dict[str, AlgoTradeBase]
) -> None:
    """
    モデルをインポートして辞書に追加（既存のインターフェース維持用）

    Args:
        name_model: モデル名
        dict_model: モデルを格納する辞書
    """
    dict_model[name_model] = get_model_instance(name_model)


# 使用例
if __name__ == "__main__":
    # パターンA: インスタンスを直接取得（推奨）
    model = get_model_instance("algo_trade")
    print(f"Loaded: {model.getName()} v{model.getVersion()}")

    # パターンB: 辞書に追加（既存互換）
    models = {}
    load_model_into_dict("algo_trade", models)
    print(f"Loaded: {models['algo_trade'].getName()}")