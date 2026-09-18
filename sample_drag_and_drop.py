from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)

from structs.file_path import (
    FilePathModel,
    FilePathProxyModel,
)
from widgets.listviews import ListViewFileDnD


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("QListView Sample")

        self.model = model = FilePathModel()
        lv = ListViewFileDnD(model, self)

        proxy = FilePathProxyModel()
        proxy.setDynamicSortFilter(True)
        proxy.setSourceModel(model)
        proxy.sort(0, Qt.SortOrder.AscendingOrder)

        lv.setModel(proxy)

        layout = QVBoxLayout()
        layout.addWidget(lv)

        widget = QWidget()
        widget.setLayout(layout)

        self.setCentralWidget(widget)


if __name__ == "__main__":
    app = QApplication([])

    window = MainWindow()
    window.resize(400, 300)
    window.show()

    app.exec()
