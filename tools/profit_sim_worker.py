from PySide6.QtCore import QObject, Signal, Slot

class PluginWorker(QObject):

    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, plugin):
        super().__init__()
        self.plugin = plugin

    @Slot()
    def run(self):
        try:
            dict_result = self.plugin.run()
            self.finished.emit(dict_result)
        except Exception as e:
            self.error.emit(str(e))