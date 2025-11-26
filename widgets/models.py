import math
from typing import Any

import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt


class ModelTransaction(QAbstractTableModel):
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

        Return list_data cell from the pandas DataFrame
        """
        if not index.isValid():
            return None

        row = index.row()
        col = index.column()
        value = self._dataframe.iloc[row, col]
        if (type(value) is int) | (type(value) is float):
            if math.isnan(value):
                value = ''

        if role == Qt.ItemDataRole.DisplayRole:
            return str(value)
        elif role == Qt.ItemDataRole.TextAlignmentRole:
            if col == 1:
                # 注文日時
                flag = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            elif col == 2:
                # 銘柄コード
                flag = Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
            elif col == 3:
                # 売買
                flag = Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
            elif (type(value) is int) | (type(value) is float):
                flag = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            else:
                flag = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            return flag

        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = ...) -> Any:
        """Override method from QAbstractTableModel

        Return dataframe index as vertical header list_data and columns as horizontal header list_data.
        """
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._dataframe.columns[section])

            if orientation == Qt.Orientation.Vertical:
                # return str(self._dataframe.index[section])
                return None

        return None
