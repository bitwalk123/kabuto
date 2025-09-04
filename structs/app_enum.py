from enum import Enum, auto


class FollowType(Enum):
    PARABOLIC = auto()  # Parabolic SAR の設定によるトレンドフォロー
    OVERDRIVE = auto()  # 価格と PSAR の値幅が大きくなったので追跡フォロー
    BEP = auto()  # 損益分岐点 (Break-Even Point) に近づけるフォロー
    DECELERATE = auto()  # 追跡フォロー後の減速段階


class PositionType(Enum):
    BUY = auto()
    SELL = auto()
    REPAY = auto()
    HOLD = auto()
    NONE = auto()


class TransactionType(Enum):
    BUY = auto()
    SELL = auto()
    REPAY = auto()
    HOLD = auto()
