import os

from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QIcon

from modules.kabuto import Kabuto
from modules.simulator import SimulatorWorker
from modules.trend_charts import SimulationCharts
from structs.file_path import FilePath
from structs.res import AppRes
from widgets.containers import MainWindow
from widgets.docks import DockFileList, DockSimulation
from widgets.statusbars import StatusBar
from widgets.toolbars import ToolBarBeetle


class Beetle(MainWindow):
    __app_name__ = "Beetle"
    __version__ = Kabuto.__version__
    __author__ = "Fuhito Suguri"
    __license__ = "MIT"

    thread: QThread
    worker: SimulatorWorker

    def __init__(self):
        super().__init__()
        self.res = res = AppRes()

        # ウィンドウアイコンとタイトルを設定
        self.setWindowIcon(QIcon(os.path.join(self.res.dir_image, "beetle.png")))
        title_win = f"{self.__app_name__} - {self.__version__}"
        self.setWindowTitle(title_win)

        # ツール・バー
        toolbar = ToolBarBeetle(res)
        self.addToolBar(toolbar)

        # 左ドック
        self.dock_files = dock_files = DockFileList()
        self.addDockWidget(
            Qt.DockWidgetArea.LeftDockWidgetArea,
            dock_files
        )

        # 右ドック
        self.dock_sim = dock_sim = DockSimulation()
        self.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea,
            dock_sim
        )
        dock_sim.clickedStart.connect(self.on_start)

        chart = SimulationCharts(res)
        self.setCentralWidget(chart)

        status = StatusBar(res)
        self.setStatusBar(status)

    def on_start(self):
        list_files: list[FilePath] = self.dock_files.get_files()
        if len(list_files) == 0:
            return

        """
        for obj_file in list_files:
            print(obj_file.name)
        """
        obj_file = list_files[-1]
        self.dock_files.select_file(obj_file)
        self.do_simulation(obj_file)

    def do_simulation(self, obj_file: FilePath):
        """
        別スレッドでシミュレーションを実行
        :param obj_file:
        """
        self.thread = thread = QThread()
        self.worker = worker = SimulatorWorker(obj_file)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        thread.finished.connect(self.next_simulation)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        worker.result.connect(self.handle_result)

        thread.start()

    def handle_result(self, result: dict):
        print(result)

    def next_simulation(self):
        pass
