import logging

from PySide6.QtCore import QObject, Signal, Slot

from modules.rl_ppo_lite_20250901 import TradingSimulation


class RLModelWorker(QObject):
    # 売買アクションを通知
    notifyAction = Signal(str)
    finished = Signal()

    def __init__(self, autopilot: bool):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.autopilot = autopilot
        self._running = True
        self._stop_flag = False

        # シミュレータ・インスタンス
        model_path = "models/ppo_7011_20250829.pch"
        self.sim = TradingSimulation(model_path)

    @Slot(float, float, float)
    def addData(self, ts, price, volume, force_close=False):
        action = self.sim.add(ts, price, volume, force_close=force_close)
        if self.autopilot:
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # 🧿 売買アクションを通知するシグナル
            self.notifyAction.emit(action)
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    @Slot(bool)
    def setAutoPilotStatus(self, state: bool):
        self.autopilot = state
        self.logger.info(f"{__name__}: autopilot is set to {state}.")

    @Slot()
    def stop(self):
        """終了処理"""
        self._stop_flag = True
        self.finished.emit()
