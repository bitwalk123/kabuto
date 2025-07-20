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
        self.has_position = False

    def setTrend(self, trend: int, epupd: int) -> PositionType:
        self.counter += 1
        self.epupd = epupd
        if self.trend != trend:
            if self.trend * trend == -1:
                ptype = PositionType.REPAY
                self.has_position = False
            else:
                ptype = PositionType.NONE
            self.trend = trend
        else:
            if self.num < epupd and not self.has_position:
                if self.trend == 1:
                    ptype = PositionType.BUY
                    self.has_position = True
                elif self.trend == -1:
                    ptype = PositionType.SELL
                    self.has_position = True
                else:
                    ptype = PositionType.NONE
            else:
                ptype = PositionType.NONE
        return ptype
