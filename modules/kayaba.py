import datetime
import glob
import logging
import os
from typing import Any, Literal, TypeAlias

import pandas as pd

from funcs.commons import (
    check_doe_factor_match,
    get_datestr_from_collections,
    get_dt_from_collections,
)
from funcs.excel import is_sheet_exists, load_excel
from funcs.setting import load_setting, update_setting
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

    def __init__(self, name_doe: str, code: str, dt_start: datetime.datetime) -> None:
        self.logger = logging.getLogger(__name__)
        self.name_doe = name_doe
        self.code = code
        self.dt_start = dt_start
        self.res = res = AppRes()

        path_csv = os.path.join(res.dir_doe, f"{name_doe}.csv")
        self.df_doe = pd.read_csv(path_csv)

        # 最新のパラメータへ更新
        update_setting(res, code)
        # パラメータの読み込み
        self.dict_setting: dict[str, Any] = load_setting(res, code)

        # 実験表と設定ファイルを比較、因子が一致していることが前提
        if not check_doe_factor_match(self.df_doe, self.dict_setting):
            self.logger.error("DOE の因子と設定ファイルの因子が一致しませんでした。")
            raise

        print("下記の条件で DOE を実施します。")
        print(self.df_doe)

        # ポジション・マネージャ（使い回す）
        self.posman = PositionManager()

    def run(self) -> None:
        self.logger.info(self.name_doe)
        days = 0  # 日数
        grand_total = 0  # 総収益

        # ティックファイル
        path_glob = os.path.join(self.res.dir_collection, f"*.xlsx")
        list_excel = sorted(glob.glob(path_glob))
        for path_excel in list_excel:
            if get_dt_from_collections(path_excel) < self.dt_start:
                continue
            if not is_sheet_exists(path_excel, self.code):
                continue

            # Excel 名
            print(f"\n{path_excel}")
            # 日付文字列
            date_str = get_datestr_from_collections(path_excel)
            # Excel シートの読み込み
            dict_sheet = load_excel(path_excel)
            # 対象シートのデータフレーム
            df: pd.DataFrame = dict_sheet[self.code]

            # シミュレーション用エージェント（パラメータ毎に生成し直す）
            agent = AgentCLI(self.code, self.dict_setting)

            # シミュレーション
            n, total = self.simulation(agent, df)
            self.logger.info(f"{date_str}: 売買回数: {n} 回, 損益: {total: .0f} 円")
            grand_total += total
            days += 1

            # テクニカル・データ（出力先）
            path_csv = os.path.join(self.res.dir_temp, f"{date_str}_{self.code}_technicals.csv")
            self.saveTechnicals(agent, path_csv)

        print(f"総収益: {grand_total} 円, {days} 日")

    def simulation(self, agent: AgentCLI, df: pd.DataFrame) -> tuple[int, Any]:
        # ポジション・マネージャのリセット
        self.posman.reset()
        # ポジション・マネージャの初期化
        self.posman.initPosition([self.code])

        # エージェントの自動売買をオン
        agent.setAutoPilot(True)
        # エージェントのリセット
        agent.resetEnv()

        # ティックデータのループ
        ts, price = (0.0, 0.0)
        for row in df.itertuples():
            ts = row.Time
            price = row.Price
            volume = row.Volume

            action, position = agent.addData(ts, price, volume)
            action_enum = ActionType(action)
            if action_enum != ActionType.HOLD:
                method_name = self.ACTION_DISPATCH.get((action_enum, position))
                if method_name is None:
                    self.logger.error(f"trade rule violation! action={action_enum}, pos={position}")
                    break
                getattr(self, method_name)(ts, price)

        # 強制返済
        self.forceRepay(ts, price)

        # 取引明細
        df_transaction = self.posman.getTransactionResult()

        return len(df_transaction), df_transaction["損益"].sum()

    def doBuy(self, ts, price) -> None:
        self.posman.openPosition(self.code, ts, price, ActionType.BUY)

    def doSell(self, ts, price) -> None:
        self.posman.openPosition(self.code, ts, price, ActionType.SELL)

    def doRepay(self, ts, price) -> None:
        self.posman.closePosition(self.code, ts, price)

    def forceRepay(self, ts, price) -> None:
        if self.posman.hasPosition(self.code):
            self.doRepay(ts, price)

    def saveTechnicals(self, agent: AgentCLI, path_csv: str) -> None:
        # テクニカル・データの保存
        agent.saveTechnicals(path_csv)
