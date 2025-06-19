from PySide6.QtWidgets import QMainWindow

from structs.res import AppRes




class StockVein(QMainWindow):
    def __init__(self):
        super().__init__()
        self.res = res = AppRes()
