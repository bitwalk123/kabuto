import logging
import os

import pandas as pd
from PySide6.QtCore import QObject, Signal, Slot
from pandas import DataFrame

from structs.file_path import FilePath


class SimulatorWorker(QObject):
    finished = Signal()
    result = Signal(dict)

    def __init__(self, obj_file: FilePath) -> None:
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.sim = Simulator(obj_file)

    @Slot()
    def run(self):
        # 重い処理
        dict_result = self.sim.start()
        self.result.emit(dict_result)
        self.finished.emit()


class Simulator():
    def __init__(self, obj_file: FilePath, code: str = "9984", full: bool = False):
        self.logger = logging.getLogger(__name__)
        self.obj_file = obj_file
        self.code = code
        self.full = full

    def start(self) -> dict:
        df = self.read_excel()
        if len(df) == 0:
            return {}

        print(self.obj_file.full)
        print(self.code)
        print(df)
        return {}

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
