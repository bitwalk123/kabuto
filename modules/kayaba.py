import logging
from typing import Any, Literal, TypeAlias

import pandas as pd

from modules.agent_cli import AgentCLI
from modules.kabuto import Kabuto
from modules.posman import PositionManager
from structs.app_enum import ActionType, PositionType
from structs.res import AppRes

# 型エイリアスの定義（クラスの外に配置）
TradeAction: TypeAlias = Literal["doBuy", "doSell", "doRepay"]
TradeKey: TypeAlias = tuple[ActionType, PositionType]


class Kayaba:
    __app_name__ = "Kayaba"
    __version__ = Kabuto.__version__
    __author__ = "Fuhito Suguri"
    __license__ = "MIT"

    # --- 状態遷移表 ---
    ACTION_DISPATCH: dict[TradeKey, TradeAction] = {
        (ActionType.BUY, PositionType.NONE): "doBuy",  # 建玉がなければ買建
        (ActionType.BUY, PositionType.SHORT): "doRepay",  # 売建（ショート）であれば（買って）返済
        (ActionType.SELL, PositionType.NONE): "doSell",  # 建玉がなければ売建
        (ActionType.SELL, PositionType.LONG): "doRepay",  # 買建（ロング）であれば（売って）返済
        # HOLD は何もしないので載せない
    }

    def __init__(self, res: AppRes, code: str, dict_setting: dict[str, Any], df: pd.DataFrame):
        self.logger = logging.getLogger(__name__)
        self.res = res
        self.code = code
        self.dict_setting = dict_setting
        self.df = df

        # ポジション・マネージャ
        self.posman = PositionManager()

        # PySide6 に依存しない GUI 無しエージェントのインスタンス
        self.agent = agent = AgentCLI(code, dict_setting)
        agent.setAutoPilot(True)

    def run(self)->float:
        # ポジション・マネージャの初期化
        self.posman.initPosition([self.code])
        # エージェントのリセット
        self.agent.resetEnv()
        # ティックデータのループ
        ts, price = (0.0, 0.0)
        for row in self.df.itertuples():
            ts = row.Time
            price = row.Price
            volume = row.Volume

            action, position = self.agent.addData(ts, price, volume)
            action_enum = ActionType(action)

            if action_enum != ActionType.HOLD:
                method_name = self.ACTION_DISPATCH.get((action_enum, position))
                if method_name is None:
                    self.logger.error(
                        f"trade rule violation! action={action_enum}, pos={position}"
                    )
                    break
                getattr(self, method_name)(ts, price)

        # 強制返済
        self.forceRepay(ts, price)
        df_transaction = self.posman.getTransactionResult()

        return df_transaction["損益"].sum()

    def doBuy(self, ts, price):
        self.posman.openPosition(self.code, ts, price, ActionType.BUY)

    def doSell(self, ts, price):
        self.posman.openPosition(self.code, ts, price, ActionType.SELL)

    def doRepay(self, ts, price):
        self.posman.closePosition(self.code, ts, price)

    def forceRepay(self, ts, price):
        if self.posman.hasPosition(self.code):
            self.doRepay(ts, price)

    def saveTechnicals(self, path_csv: str):
        # テクニカル・データの保存
        self.agent.saveTechnicals(path_csv)
