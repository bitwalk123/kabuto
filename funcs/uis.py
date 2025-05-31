from PySide6.QtWidgets import QBoxLayout


def clear_boxlayout(layout: QBoxLayout):
    for i in reversed(range(layout.count())):
        layout.itemAt(i).widget().deleteLater()
