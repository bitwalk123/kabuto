import logging

from PySide6.QtCore import QObject, Signal, Slot

from modules.rl_ppo_lite_20250821 import TradingSimulation


class RLModelWorker(QObject):
    # 売買アクションを通知
    notifyAction = Signal(str)
    finished = Signal()

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self._running = True
        self._stop_flag = False

        # シミュレータ・インスタンス
        model_path = "models/ppo_7011_20250821.pt"
        self.sim = TradingSimulation(model_path)

    @Slot(float, float, float)
    def addData(self, ts, price, volume, force_close=False):
        action = self.sim.add(ts, price, volume, force_close=force_close)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 🧿 売買アクションを通知するシグナル
        self.notifyAction.emit(action)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    @Slot()
    def stop(self):
        """終了処理"""
        self._stop_flag = True
        self.finished.emit()
