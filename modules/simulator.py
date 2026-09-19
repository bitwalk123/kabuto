import logging
import os

import pandas as pd
from PySide6.QtCore import QObject, Signal, Slot
from pandas import DataFrame

from funcs.tide import get_ts_1h_end
from modules.agent import SimulatorAgent
from modules.posman import PositionManager
from structs.file_path import FilePath


class Simulator():
    ts_1h_end: float

    def __init__(self, obj_file: FilePath, code: str = "9984", full: bool = False):
        self.logger = logging.getLogger(__name__)
        self.obj_file = obj_file
        self.code = code
        self.full = full

    def start(self) -> dict:
        dict_result = {}
        df: pd.DataFrame = self.read_excel()
        size_row = len(df)
        if size_row == 0:
            return dict_result

        print(self.obj_file.full)
        print(self.code)
        # print(df)

        # 前引け時刻
        ts = df.iloc[0]["Time"]
        self.ts_1h_end = get_ts_1h_end(ts)
        print(self.ts_1h_end)

        agent = SimulatorAgent(self.code, {})
        agent.resetEnv()
        posman = PositionManager()
        posman.initPosition([self.code])
        for r in range(size_row):
            # 一行のデータ
            row = df.iloc[r]
            ts = row["Time"]
            price = row["Price"]
            volume = row["Volume"]
            # ポジションマネージャからの含み益などの情報
            dict_info = posman.getInfo(self.code, price)
            # エージェントへ情報追加
            agent.addData(ts, price, volume, dict_info)

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
            self.logger.error(
                f"{self.obj_file.full} は存在しません。"
            )
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
