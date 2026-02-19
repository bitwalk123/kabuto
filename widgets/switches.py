from PySide6.QtCore import (
    Property,
    QByteArray,
    QPropertyAnimation,
    QRect,
    QSize,
    Qt,
    Signal,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QMouseEvent,
    QPainter,
    QPaintEvent,
)
from PySide6.QtWidgets import QAbstractButton


class Switch(QAbstractButton):
    """Implementation of a clean looking toggle disparity translated from
    https://stackoverflow.com/a/38102598/1124661
    QAbstractButton::setDisabled to disable
    """
    statusChanged = Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        self.onBrush: QBrush = QBrush(QColor("#569167"))
        self.slotBrush: QBrush = QBrush(QColor("#999999"))
        self.switchBrush: QBrush = self.slotBrush
        self.disabledBrush: QBrush = QBrush(QColor("#666666"))
        self.on: bool = False
        self.fullHeight: int = 18
        self.halfHeight: int = int(self.fullHeight / 2)
        self.xPos: int = self.halfHeight
        self.fullWidth: int = 34
        self.setFixedWidth(self.fullWidth)
        self.slotMargin: int = 3
        self.slotHeight: int = self.fullHeight - 2 * self.slotMargin
        self.travel: int = self.fullWidth - self.fullHeight
        self.slotRect: QRect = QRect(
            self.slotMargin,
            self.slotMargin,
            self.fullWidth - 2 * self.slotMargin,
            self.slotHeight,
        )
        self.animation: QPropertyAnimation = QPropertyAnimation(self, QByteArray(b'pqProp'), self)
        self.animation.setDuration(120)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def paintEvent(self, e: QPaintEvent) -> None:
        """QAbstractButton method. Paint the button."""
        painter = QPainter(self)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self.switchBrush if self.on else self.disabledBrush)
        painter.setOpacity(0.6)
        painter.drawRoundedRect(
            self.slotRect, self.slotHeight / 2, self.slotHeight / 2,
        )
        painter.setOpacity(1.0)
        painter.drawEllipse(
            QRect(self.xPos, 0, self.fullHeight, self.fullHeight)
        )

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:
        """Switch the button."""
        if e.button() == Qt.MouseButton.LeftButton:
            self.on = not self.on
            self.switchBrush = self.onBrush if self.on else self.slotBrush
            self.animation.setStartValue(self.xPos)
            self.animation.setEndValue(self.travel if self.on else 0)
            self.animation.start()
            self.statusChanged.emit(self.on)
        super().mouseReleaseEvent(e)

    def sizeHint(self) -> QSize:
        """Required to be implemented and return the size of the widget."""
        return QSize(self.fullWidth, self.fullHeight)

    def setOffset(self, o: int) -> None:
        """Setter for QPropertyAnimation."""
        self.xPos = o
        self.update()

    def getOffset(self) -> int:
        """Getter for QPropertyAnimation."""
        return self.xPos

    pqProp = Property(int, fget=getOffset, fset=setOffset)  # type: ignore[arg-type]

    def set(self, on: bool) -> None:
        """Set state to on, and trigger repaint."""
        self.on = on
        self.switchBrush = self.onBrush if on else self.slotBrush
        self.xPos = self.travel if on else 0
        self.update()

    def isEnabled(self) -> bool:
        return self.on
