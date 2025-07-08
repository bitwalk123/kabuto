from collections import deque  # deque をインポート


class PSARObject:
    def __init__(self):
        self.price: float = 0.
        self.y: float = 0.
        self.trend: int = 0
        self.ep: float = 0.
        self.af: float = -1.  # AF は 0 以上の実数
        self.psar: float = 0.
        self.epupd: int = 0
        self.duration: int = 0
        self.distance: float = 0


class RealtimePSAR:
    def __init__(
            self,
            af_init: float = 0.00002,
            af_step: float = 0.00002,
            af_max: float = 0.002,
            rolling_n: int = 60
    ):
        self.af_init = af_init
        self.af_step = af_step
        self.af_max = af_max

        # PSARObject のインスタンス
        self.obj = PSARObject()

        n_smoothing = 600
        self.xs_deque = deque(maxlen=n_smoothing)
        self.prices_deque = deque(maxlen=n_smoothing)

    def add(self, price: float) -> PSARObject:
        if self.obj.trend == 0:
            pass
        elif self.cmp_psar(price):
            # トレンド反転
            self.obj.price = price
            self.obj.trend *= -1
            self.obj.psar = self.obj.ep
            self.obj.ep = price
            self.obj.af = self.af_init
            self.obj.epupd = 0
            self.obj.duration = 0
            self.obj.distance = abs(price - self.obj.psar)
            self.first_trend = False  # 最初のトレンドフラグを False に
            # return self.obj
        else:
            # トレンド維持
            if self.cmp_ep(price):
                self.update_ep_af(price)

            # PSAR の更新
            self.obj.psar = self.obj.psar + self.obj.af * (self.obj.ep - self.obj.psar)

            # 最初のトレンドのみの対応
            # if self.first_trend:
            #    self.trend_follow_aggressive(price)

            self.obj.price = price
            self.obj.duration += 1
            # return self.obj

        return self.obj

    def cmp_ep(self, price: float) -> bool:
        if 0 < self.obj.trend:
            if self.obj.ep < price:
                return True
            else:
                return False
        else:
            if price < self.obj.ep:
                return True
            else:
                return False

    def cmp_psar(self, price: float) -> bool:
        if 0 < self.obj.trend:
            if price < self.obj.psar:
                return True
            else:
                return False
        else:
            if self.obj.psar < price:
                return True
            else:
                return False

    def update_ep_af(self, price: float):
        self.obj.ep = price
        self.obj.epupd += 1
        self.obj.duration = 0
        if self.obj.af < self.af_max - self.af_step:
            self.obj.af += self.af_step
