from typing import Any

import numpy as np
import pandas as pd
from PySide6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    Qt,
)
from PySide6.QtWidgets import (
    QHeaderView,
    QMainWindow,
    QTableView,
)

from structs.res import AppRes


class PandasModel(QAbstractTableModel):
    """A model to interface a Qt view with pandas dataframe """

    def __init__(self, dataframe: pd.DataFrame, parent=None):
        QAbstractTableModel.__init__(self, parent)
        self._dataframe = dataframe

    def rowCount(self, parent=QModelIndex()) -> int:
        """ Override method from QAbstractTableModel

        Return row count of the pandas DataFrame
        """
        if parent == QModelIndex():
            return len(self._dataframe)
        return 0

    def columnCount(self, parent=QModelIndex()) -> int:
        """Override method from QAbstractTableModel

        Return column count of the pandas DataFrame
        """
        if parent == QModelIndex():
            return len(self._dataframe.columns)
        return 0

    def data(self, index: QModelIndex, role=Qt.ItemDataRole):
        """Override method from QAbstractTableModel

        Return data cell from the pandas DataFrame
        """
        if not index.isValid():
            return None

        row = index.row()
        col = index.column()
        value = self._dataframe.iloc[row, col]

        if role == Qt.ItemDataRole.DisplayRole:
            return str(value)
        elif role == Qt.ItemDataRole.TextAlignmentRole:
            if (type(value) is np.int64) | (type(value) is np.float64):
                flag = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            else:
                flag = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            return flag

        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = ...) -> Any:
        """Override method from QAbstractTableModel

        Return dataframe index as vertical header data and columns as horizontal header data.
        """
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._dataframe.columns[section])

            if orientation == Qt.Orientation.Vertical:
                # return str(self._dataframe.index[section])
                return None

        return None


class WinTransaction(QMainWindow):
    def __init__(self, res: AppRes, df: pd.DataFrame):
        super().__init__()
        self.res = res
        self.setWindowTitle("取引履歴")

        view = QTableView()
        self.setCentralWidget(view)

        view.setAlternatingRowColors(True)
        # view.horizontalHeader().setStretchLastSection(True)
        # view.setSelectionBehavior(QTableView.SelectRows)

        model = PandasModel(df)
        view.setModel(model)

        header = view.horizontalHeader()
        header.setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
