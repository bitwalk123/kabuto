import logging

import pandas as pd
from PySide6.QtCore import QObject, Signal, Slot

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
    def __init__(self, obj_file: FilePath):
        self.obj_file = obj_file
        self.df = pd.read_excel(obj_file.full)

    def start(self) -> dict:
        print(self.obj_file.full)
        print(self.df)
        return {}
