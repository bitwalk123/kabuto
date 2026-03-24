import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, DefaultDict

import numpy as np
import pandas as pd

from funcs.plugin import get_model_instance
from funcs.tide import conv_datetime_from_timestamp
from modules.env import TradingEnv


class AgentBase(ABC):
    """エージェントの抽象クラス"""
    BASE_COLUMNS = ["Timestamp", "Price", "Volume"]

    def __init__(self, code: str, dict_setting: dict[str, Any]) -> None:
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.obs: np.ndarray | None = None
        self.done: bool = False
        self.df_obs: pd.DataFrame | None = None
        self._is_stopping: bool = False  # 終了フラグを追加

        # 学習環境の取得
        self.env: TradingEnv = TradingEnv(code, dict_setting)

        # モデルに渡す観測値のリスト
        self.list_obs_label: list[str] = []

        # モデルのインスタンス（とりあえずプラグイン化）
        name_model = "default"
        dict_model = {}
        get_model_instance(name_model, dict_model)
        self.model = dict_model[name_model]

        # 取引内容（＋テクニカル指標）
        self.dict_list_tech: DefaultDict[str, list[Any]] = defaultdict(list)

    @abstractmethod
    def addData(self, ts: float, price: float, volume: float) -> None:
        ...

    @abstractmethod
    def cleanup(self) -> None:
        ...

    @abstractmethod
    def forceRepay(self) -> None:
        ...

    def getTechnicals(self) -> pd.DataFrame:
        # テクニカル・データが保持されている辞書をデータフレームへ変換
        df = pd.DataFrame(self.dict_list_tech)

        # インデックスを日付形式に変換
        df.index = [pd.to_datetime(conv_datetime_from_timestamp(ts)) for ts in df["ts"]]

        return df

    @abstractmethod
    def resetEnv(self) -> None:
        ...

    def saveTechnicals(self, path_csv: str) -> pd.DataFrame | None:
        """
        テクニカル指標を CSV ファイルに保存
        :param path_csv: 保存先のファイルパス
        """
        try:
            df = self.getTechnicals()
            # 指定されたパスにデータフレームを CSV 形式で保存
            df.to_csv(path_csv)
            self.logger.info(f"{path_csv} を保存しました。")
            return df
        except (KeyError, ValueError, IOError) as e:
            self.logger.error(f"テクニカル指標の保存に失敗しました: {e}")

    @abstractmethod
    def setAutoPilot(self, flag: bool):
        ...

    @abstractmethod
    def isAutoPilot(self) -> bool:
        ...
