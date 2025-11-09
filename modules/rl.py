import logging

from PySide6.QtCore import QObject, Signal, Slot


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
        model_path = "policy.pth"
        self.sim = TradingSimulator(model_path)

    @Slot(float, float, float)
    def addData(self, ts, price, volume):
        action = self.sim.add(ts, price, volume)
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
