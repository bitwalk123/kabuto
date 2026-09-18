from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import (
    QDragEnterEvent,
    QDropEvent,
    QStandardItem,
    QStandardItemModel,
)
from PySide6.QtWidgets import QListView

from structs.file_path import FilePathModel, FilePath


class CheckList(QListView):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("QListView {font-family: monospace;}")
        self.setAlternatingRowColors(True)
        self.model = model = QStandardItemModel(self)
        self.setModel(model)

    def addItems(self, list_item: list, row_default: int = 0):
        for item in list_item:
            item = QStandardItem(item)
            item.setCheckable(True)
            self.model.appendRow(item)
        # デフォルト銘柄の選択
        self.model.item(row_default).setCheckState(Qt.CheckState.Checked)

    def getSelected(self) -> list:
        """
        チェックが入った行番号をリストで返す
        :return:
        """
        return [
            row
            for row in range(self.model.rowCount())
            if self.model.item(row).checkState() == Qt.CheckState.Checked
        ]


class ListViewFileDnD(QListView):
    def __init__(self, model: FilePathModel, parent=None):
        super().__init__(parent)

        self.model = model
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        if not event.mimeData().hasUrls():
            event.ignore()
            return

        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())

            if path.is_file():
                self.model.add_file(FilePath(path))

        event.acceptProposedAction()
