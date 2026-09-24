import logging
import os
from typing import Any

import pandas as pd
from PySide6.QtCore import QObject, Signal, Slot
from pandas import DataFrame

from funcs.tide import get_dt_market_range, get_ts_trade_end
from funcs.tse import get_ticker_name_list
from modules.agent import SimulatorAgent
from modules.posman import PositionManager
from structs.app_enum import ActionType, PositionType
from structs.file_path import FilePath


class Simulator():
    agent: SimulatorAgent

    def __init__(
            self,
            obj_file: FilePath,
            dict_option: dict,
    ):
        self.logger = logging.getLogger(__name__)
        self.obj_file = obj_file
        if "code" in dict_option:
            self.code = dict_option["code"]
        else:
            self.code = "0000"
        self.dict_setting = {}
        self.dict_option = dict_option

        # ポジション・マネージャ
        self.posman = posman = PositionManager()
        posman.initPosition([self.code])

    def start(self) -> dict:
        df: pd.DataFrame = self.read_excel()
        size_row = len(df)
        if size_row == 0:
            return {}

        """
        print(self.obj_file.full)
        print(self.code)
        """
        ts0 = df.iloc[0]["Time"]
        ts_end = get_ts_trade_end(ts0)

        # 結果格納用辞書準備
        dict_result = self.prep_dict_result(ts0)

        # シミュレーション用エージェントのインスタンス
        self.agent = agent = SimulatorAgent(self.code, self.dict_setting)
        # 環境のリセット
        agent.resetEnv()
        # 環境オプションの設定
        self.set_env_options()

        # ティックデータのループ
        ts = 0
        price = 0
        for r in range(size_row):
            # 一行のデータ
            row = df.iloc[r]
            ts = row["Time"]
            price = row["Price"]
            volume = row["Volume"]
            # ポジションマネージャからの含み益などの情報
            dict_info = self.posman.getInfo(self.code, price)

            # エージェントへ情報追加
            action, position, states = agent.addData(ts, price, volume, dict_info)
            action_type = ActionType(action)
            if "reason" in states:
                note = states["reason"]
            else:
                note = ""
            if action_type != ActionType.HOLD:
                if position == PositionType.NONE:
                    if action_type == ActionType.BUY:
                        # 買建
                        self.posman.openPosition(self.code, ts, price, ActionType.BUY, note)
                    elif action_type == ActionType.SELL:
                        # 売建
                        self.posman.openPosition(self.code, ts, price, ActionType.SELL, note)
                else:
                    # 返済
                    self.posman.closePosition(self.code, ts, price, note)

        # 終了処理（ティックデータ最後のデータは 15:24:50 直前）
        position = agent.env.getCurrentPosition()
        if position != PositionType.NONE:
            agent.forceRepay()
            # 返済
            note = "強制返済"
            self.posman.closePosition(self.code, ts, price, note)
            self.logger.info(f"'{self.code}'の強制返済をしました。")

        # 取引明細
        dict_result["transaction"] = self.posman.getTransactionResult()
        # テクニカルデータのデータフレーム
        dict_result["technicals"] = agent.getTechnicals()

        return dict_result

    def prep_dict_result(self, ts: float) -> dict[str, Any]:
        dict_result = {}
        # 取引時間
        dt_start, dt_end = get_dt_market_range(ts)
        dict_result["dt_open"] = dt_start
        dict_result["dt_close"] = dt_end

        # 銘柄名 (銘柄コード)
        name = get_ticker_name_list([self.code])[self.code]
        dict_result["title"] = f"{name} ({self.code})"
        return dict_result

    def read_excel(self) -> DataFrame:
        # 指定した銘柄コード self.code のシートを読み込む
        if os.path.exists(self.obj_file.full):
            wb = pd.ExcelFile(self.obj_file.full)
            # Excel シートの一覧
            list_sheet: list = wb.sheet_names
            if self.code in list_sheet:
                return wb.parse(sheet_name=self.code)
            else:
                self.logger.error(
                    f"{self.obj_file.full} にシート {self.code} が存在しません。"
                )
                return pd.DataFrame()
        else:
            self.logger.error(f"{self.obj_file.full} は存在しません。")
            return pd.DataFrame()

    def set_env_options(self):
        # 環境オプションの設定
        if "cross_ma" in self.dict_option:
            self.agent.updateStateCrossMA(self.dict_option["cross_ma"])
        if "cross_vwap" in self.dict_option:
            self.agent.updateStateCrossVWAP(self.dict_option["cross_vwap"])
        if "profit_vwap" in self.dict_option:
            self.agent.updateStateProfitVWAP(self.dict_option["profit_vwap"])
        if "losscut_vwap" in self.dict_option:
            self.agent.updateStateLosscutVWAP(self.dict_option["losscut_vwap"])


class SimulatorWorker(QObject):
    finished = Signal()
    result = Signal(dict)

    def __init__(self, obj_file: FilePath, dict_option: dict) -> None:
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.sim = Simulator(obj_file, dict_option)

    @Slot()
    def run(self):
        self.logger.info("シミュレーションを開始します。")
        dict_result = self.sim.start()
        self.logger.info("シミュレーションが終了しました。")

        self.result.emit(dict_result)
        self.finished.emit()
