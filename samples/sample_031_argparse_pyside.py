import sys

from PySide6.QtWidgets import QApplication, QWidget


class Example(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

def main():
    app = QApplication(sys.argv)
    win = Example()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
