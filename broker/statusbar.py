from PySide6.QtCore import Signal
from PySide6.QtWidgets import QStatusBar, QLineEdit, QLabel

from structs.res import AppRes


class StatusBarBrokerClient(QStatusBar):
    requestSendMessage = Signal(str)

    def __init__(self, res: AppRes):
        super().__init__()
        self.res = res

        lab_message = QLabel("Message")
        self.addWidget(lab_message)

        self.ledit = ledit = QLineEdit(self)
        ledit.returnPressed.connect(self.send_message)  # Send when Return key is pressed
        self.addWidget(ledit, stretch=1)

    def send_message(self):
        msg = self.ledit.text()
        self.requestSendMessage.emit(msg)
        self.ledit.clear()  # Clear the input field after sending
