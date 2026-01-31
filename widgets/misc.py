from PySide6.QtGui import QFont, QFontDatabase

from structs.res import AppRes


class TickFont(QFont):
    def __init__(self, res: AppRes):
        super().__init__()
        font_id = QFontDatabase.addApplicationFont(res.path_monospace)
        font_name = QFontDatabase.applicationFontFamilies(font_id)[0]
        self.setFamilies(font_name)
        self.setPointSize(9)
