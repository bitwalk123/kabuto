from enum import Enum, auto


class PositionType(Enum):
    BUY = auto()
    SELL = auto()
    REPAY = auto()
    HOLD = auto()
    NONE = auto()
