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
        self.obj_file = obj_file

    @Slot()
    def run(self):
        # 重い処理
        #result = self.do_work()
        print(self.obj_file.full)
        df = pd.read_excel(self.obj_file.full)
        print(df)
        dict_result = {}

        self.result.emit(dict_result)
        self.finished.emit()
