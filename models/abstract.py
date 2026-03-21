from abc import ABC, abstractmethod
from typing import Any

from structs.app_enum import ActionType, PositionType


class AlgoTradeBase(ABC):
    """
    強化学習モデルの代わりに、自作のアルゴリズムで取引する疑似モデルのベース・クラス
    """
    name: str = "AlgoTradeBase"
    version: str = "1.0.0"
    MODEL_NAME: str = ""

    def __init__(self):
        self.autopilot = False
        self.list_obs_label: list | None = None

    @staticmethod
    def can_execute(action, masks):
        """
        アクションが行動マスクで禁止されていないかチェック
        :param action:
        :param masks:
        :return:
        """
        return masks[action] == 1

    @staticmethod
    def exit_action(position: PositionType):
        """
        ポジションに応じた返済アクション
        【備考】
        以前は返済アクションにREPAYMENTを用意していたが、
        強化モデルの学習に合わないので削除した。→ 疑似モデルでも維持
        :param position:
        :return:
        """
        if position == PositionType.LONG:
            return ActionType.SELL.value
        if position == PositionType.SHORT:
            return ActionType.BUY.value
        return None

    def getListObs(self) -> list:
        return self.list_obs_label

    def getName(self) -> str:
        """戦略名を返す"""
        return self.name

    def getVersion(self) -> str:
        """バージョンを返す"""
        return self.version

    @abstractmethod
    def predict(self, obs, masks) -> tuple[int, dict[str, Any]]:
        """
        観測値から行動を予測（必須実装）

        :param obs: 観測値（numpy配列）
        :param masks: 行動マスク
        :return: (action, info_dict)
        """
        ...

    def setAutoPilot(self, flag: bool):
        self.autopilot = flag

    @abstractmethod
    def updateObs(self, list_obs_label):
        """
        観測値ラベルの更新（必須実装）

        疑似ロジックでは、観測値にラベルを付けておかないと、コーディングする側が間違える！
        【課題】
        ObservationManager クラスと整合・同期を取る仕組みを導入する必要がある。

        :param list_obs_label: 観測値ラベルのリスト
        :return:
        """
        ...
