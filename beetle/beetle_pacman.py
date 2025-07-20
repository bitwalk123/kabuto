class PacMan:
    """
    トレンド判定アルゴリズム
    """

    def __init__(self):
        self.counter = 0  # トータルカウンター
        self.sar = 0  # 反転カウンター
        self.trend: int = 0  # トレンドの向き
        self.epupd: int = 0  # 更新回数

    def setTrend(self, trend: int, epupd: int) -> bool:
        self.counter += 1
        self.epupd = epupd
        if self.trend != trend:
            self.trend = trend
        return True

    def getAction(self):
        pass