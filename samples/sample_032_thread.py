import sys
from PySide6.QtWidgets import (
    QApplication,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class MyButton(QPushButton):
    def __init__(self):
        super().__init__()
        self.setText("スレッドのテスト")

class Example(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QThread test")
        layout = QVBoxLayout()
        self.setLayout(layout)

        but = MyButton()
        layout.addWidget(but)

def main():
    app = QApplication(sys.argv)
    ex = Example()
    ex.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()