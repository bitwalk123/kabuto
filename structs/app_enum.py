from enum import Enum, auto


class FollowType(Enum):
    PARABOLIC = auto()  # Parabolic SAR の設定によるトレンドフォロー
    CHASE = auto()  # 価格と PSAR の値幅が大きくなったので追跡フォロー
    DECELERATE = auto()  # 追跡フォロー後の減速段階


class PositionType(Enum):
    BUY = auto()
    SELL = auto()
    REPAY = auto()
    HOLD = auto()
    NONE = auto()
