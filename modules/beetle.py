import logging
import os

from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QIcon

from modules.kabuto import Kabuto
from modules.simulator import SimulatorWorker
from modules.simulator_charts import ChartWindow
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
        self.logger = logging.getLogger(__name__)
        self.res = res = AppRes()

        # ウィンドウアイコンとタイトルを設定
        self.setWindowIcon(QIcon(os.path.join(self.res.dir_image, "beetle.png")))
        title_win = f"{self.__app_name__} - {self.__version__}"
        self.setWindowTitle(title_win)

        # ツール・バー
        toolbar = ToolBarBeetle(res)
        self.addToolBar(toolbar)

        self.chart_win = chart_win = ChartWindow(res)
        self.setCentralWidget(chart_win)

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

        status = StatusBar(res)
        self.setStatusBar(status)

    def on_start(self, dict_option: dict):
        list_files: list[FilePath] = self.dock_files.get_files()
        if len(list_files) == 0:
            return

        """
        for obj_file in list_files:
            print(obj_file.name)
        """
        # 現時点では最新のデータのみ
        obj_file = list_files[-1]
        self.dock_files.select_file(obj_file)
        self.simulation_start(obj_file, dict_option)

    def simulation_start(self, obj_file: FilePath, dict_option: dict):
        """
        別スレッドでシミュレーションを実行
        :param obj_file:
        """
        self.chart_win.remove_axes()
        self.logger.info("チャートを消去しました。")

        self.thread = thread = QThread()
        self.worker = worker = SimulatorWorker(obj_file, dict_option)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.result.connect(self.simulation_done)

        worker.finished.connect(thread.quit)
        thread.finished.connect(self.simulation_next)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        thread.start()

    def simulation_done(self, dict_result: dict):
        """
        シミュレーション結果
        :param dict_result:
        :return:
        """
        if "technicals" in dict_result:
            # チャート
            self.chart_win.plot(dict_result)

        if "transaction" in dict_result:
            # 取引明細
            df_transaction = dict_result["transaction"]
            print(df_transaction)
            total = df_transaction["損益"].sum()
            print(f"合計損益: {int(total * 100)} 円（100株）")

    def simulation_next(self):
        """
        次のシミュレーション
        :return:
        """
        pass
