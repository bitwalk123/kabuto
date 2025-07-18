import sys

from PySide6.QtWidgets import (
    QApplication,
)

from samples.sample_029_dock_template import Example


def main():
    app = QApplication(sys.argv)
    win = Example()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
