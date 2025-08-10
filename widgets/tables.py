from PySide6.QtWidgets import QTableView, QHeaderView


class TransactionView(QTableView):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("""
            QTableView {
                font-family: monospace;
            }
        """)
        self.setAlternatingRowColors(True)
        self.horizontalHeader().setStretchLastSection(True)

        header = self.horizontalHeader()
        header.setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
