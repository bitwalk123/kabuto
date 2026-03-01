from PySide6.QtGui import QFont, QFontDatabase

class TickFont(QFont):
    def __init__(self, path_font:str):
        super().__init__()
        font_id = QFontDatabase.addApplicationFont(path_font)
        self.name = name = QFontDatabase.applicationFontFamilies(font_id)[0]
        self.setFamilies(name)
        self.setPointSize(9)
