import logging
import os

import pandas as pd
from PySide6.QtCore import QObject, Signal, Slot
from pandas import DataFrame

from funcs.tide import get_dt_market_range
from funcs.tse import get_ticker_name_list
from modules.agent import SimulatorAgent
from modules.posman import PositionManager
from structs.app_enum import ActionType, PositionType
from structs.file_path import FilePath


class Simulator():
    def __init__(self, obj_file: FilePath, code: str = "9984", full: bool = False):
        self.logger = logging.getLogger(__name__)
        self.obj_file = obj_file
        self.code = code
        self.full = full

        # ポジション・マネージャ
        self.posman = posman = PositionManager()
        posman.initPosition([self.code])

    def start(self) -> dict:
        dict_result = {}
        df: pd.DataFrame = self.read_excel()
        size_row = len(df)
        if size_row == 0:
            return dict_result

        """
        print(self.obj_file.full)
        print(self.code)
        """
        # 銘柄名 (銘柄コード)
        dict_result["title"] = f"{get_ticker_name_list([self.code])[self.code]} ({self.code})"

        # 取引時間
        ts = df.iloc[0]["Time"]
        dt_start, dt_end = get_dt_market_range(ts)
        dict_result["mkt_start"] = dt_start
        dict_result["mkt_end"] = dt_end

        # シミュレーション用エージェントのインスタンス
        agent = SimulatorAgent(self.code, {})
        agent.resetEnv()

        # ティックデータのループ
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

        # テクニカルデータのデータフレーム
        dict_result["technicals"] = agent.getTechnicals()
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


class SimulatorWorker(QObject):
    finished = Signal()
    result = Signal(dict)

    def __init__(self, obj_file: FilePath) -> None:
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.sim = Simulator(obj_file)

    @Slot()
    def run(self):
        self.logger.info("シミュレーションを開始します。")
        dict_result = self.sim.start()
        self.logger.info("シミュレーションが終了しました。")

        self.result.emit(dict_result)
        self.finished.emit()
