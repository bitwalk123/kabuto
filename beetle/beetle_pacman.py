from structs.posman import PositionType


class PacMan:
    """
    トレンド判定アルゴリズム
    """

    def __init__(self):
        self.counter = 0  # トータルカウンター
        self.sar = 0  # 反転カウンター
        self.trend: int = 0  # トレンドの向き
        self.epupd: int = 0  # 更新回数
        self.num = 1

    def setTrend(self, trend: int, epupd: int) -> PositionType:
        self.counter += 1
        self.epupd = epupd
        if self.trend != trend:
            if self.trend * trend == -1:
                ptype = PositionType.REPAY
            else:
                ptype = PositionType.NONE
            self.trend = trend
        else:
            if epupd == self.num:
                if self.trend == 1:
                    ptype = PositionType.BUY
                elif self.trend == -1:
                    ptype = PositionType.SELL
                else:
                    ptype = PositionType.NONE
            else:
                ptype = PositionType.NONE
        return ptype
