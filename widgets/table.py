from PySide6.QtWidgets import QTableView


class TransactionView(QTableView):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("""
            QTableView {
                font-family: monospace;
            }
        """)
        self.setAlternatingRowColors(True)
