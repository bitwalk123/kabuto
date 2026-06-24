from PySide6.QtCore import QObject, Signal, Slot

class PluginWorker(QObject):

    error = Signal(str)
    finished = Signal(dict)
    progress = Signal(int)

    def __init__(self, plugin):
        super().__init__()
        self.plugin = plugin

    @Slot()
    def run(self):
        try:
            dict_result = self.plugin.run(progress_callback=self.progress.emit)
            self.finished.emit(dict_result)
        except Exception as e:
            self.error.emit(str(e))