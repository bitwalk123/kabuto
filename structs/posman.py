from enum import Enum, auto


class PositionType(Enum):
    BUY = auto()
    SELL = auto()
    REPAY = auto()
    NONE = auto()
