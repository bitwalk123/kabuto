from enum import Enum, auto


class FollowType(Enum):
    PARABOLIC = auto()  # Parabolic SAR の設定によるトレンドフォロー
    OVERDRIVE = auto()  # 価格と PSAR の値幅が大きくなったので追跡フォロー
    BEP = auto()  # 損益分岐点 (Break-Even Point) に近づけるフォロー
    DECELERATE = auto()  # 追跡フォロー後の減速段階


class ActionType(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2


class AppMode(Enum):
    SINGLE = auto()
    ALL = auto()
    DOE = auto()


class PositionType(Enum):
    SHORT = -1
    NONE = 0
    LONG = 1


class SignalSign(Enum):
    NEGATIVE = -1
    ZERO = 0
    POSITIVE = 1


class TakeProfit(Enum):
    YES = 1
    NO = 0
