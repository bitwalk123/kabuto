from enum import Enum, auto


class FollowType(Enum):
    PARABOLIC = auto()  # Parabolic SAR の設定によるトレンドフォロー
    OVERDRIVE = auto()  # 価格と PSAR の値幅が大きくなったので追跡フォロー
    BEP = auto()  # 損益分岐点 (Break-Even Point) に近づけるフォロー
    DECELERATE = auto()  # 追跡フォロー後の減速段階

"""
class ActionType(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2
    REPAY = 3
"""


class ActionType(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2


class PositionType(Enum):
    NONE = 0
    LONG = 1
    SHORT = 2
