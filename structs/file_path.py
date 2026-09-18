from pathlib import Path
from typing import cast

from PySide6.QtCore import (
    QAbstractListModel,
    QModelIndex,
    QSortFilterProxyModel,
    Qt,
)


class FilePath:
    def __init__(self, path: str | Path):
        self.full = Path(path)
        self.name = self.full.name
        self.dir = self.full.parent
        self.ext = self.full.suffix


class FilePathModel(QAbstractListModel):
    def __init__(self):
        super().__init__()
        self.files: list[FilePath] = []
        self.paths: set[Path] = set()
        self.checked: set[Path] = set()

    def rowCount(self, parent=QModelIndex()):
        return len(self.files)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None

        file = self.files[index.row()]

        if role == Qt.ItemDataRole.DisplayRole:
            return file.name

        if role == Qt.ItemDataRole.CheckStateRole:
            return (
                Qt.CheckState.Checked
                if file.full in self.checked
                else Qt.CheckState.Unchecked
            )

        return None

    def flags(self, index):
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags

        return (
                Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsUserCheckable
        )

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        if role != Qt.ItemDataRole.CheckStateRole:
            return False

        file = self.files[index.row()]
        state = Qt.CheckState(value)

        if state == Qt.CheckState.Checked:
            self.checked.add(file.full)
        else:
            self.checked.discard(file.full)

        self.dataChanged.emit(
            index,
            index,
            [Qt.ItemDataRole.CheckStateRole],
        )
        return True

    def file_at(self, row: int) -> FilePath:
        return self.files[row]

    def add_file(self, file: FilePath):
        if file.ext.lower() != ".xlsx":
            return

        if file.full in self.paths:
            return

        row = len(self.files)

        self.beginInsertRows(QModelIndex(), row, row)
        self.files.append(file)
        self.paths.add(file.full)
        self.endInsertRows()

    def checked_files(self) -> list[FilePath]:
        return [
            file
            for file in self.files
            if file.full in self.checked
        ]

    def set_all_checked(self, checked: bool):
        if checked:
            self.checked = {file.full for file in self.files}
        else:
            self.checked.clear()

        if self.files:
            self.dataChanged.emit(
                self.index(0, 0),
                self.index(len(self.files) - 1, 0),
                [Qt.ItemDataRole.CheckStateRole],
            )


class FilePathProxyModel(QSortFilterProxyModel):
    def lessThan(self, left, right):
        model = cast(FilePathModel, self.sourceModel())

        left_file = model.file_at(left.row())
        right_file = model.file_at(right.row())

        return left_file.name.casefold() < right_file.name.casefold()

    def files(self) -> list[FilePath]:
        model = cast(FilePathModel, self.sourceModel())

        return [
            model.file_at(self.mapToSource(self.index(row, 0)).row())
            for row in range(self.rowCount())
        ]
