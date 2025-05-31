class PSARObject:
    def __init__(self):
        self.price: float = 0.
        self.trend: int = 0
        self.ep: float = 0.
        self.af: float = -1.  # AF は 0 以上の実数
        self.psar: float = 0.


class RealtimePSAR:
    def __init__(self, af_init=0.0001, af_step=0.0001, af_max=0.01):
        self.af_init = af_init
        self.af_step = af_step
        self.af_max = af_max

        self.obj = PSARObject()

    def add(self, price: float) -> PSARObject:
        if self.obj.price == 0:
            # 最初は trend = 0 に設定、price は保持
            self.obj.price = price
            self.obj.trend = 0
            return self.obj
        elif self.obj.trend == 0:
            # trend = 0 の時
            if self.obj.price < price:
                self.obj.trend = +1
                self.obj.ep = price
                self.obj.af = self.af_init
                self.obj.psar = self.obj.price
                # 保持する株価を更新（順番依存）
                self.obj.price = price
                return self.obj
            elif price < self.obj.price:
                self.obj.trend = -1
                self.obj.ep = price
                self.obj.af = self.af_init
                self.obj.psar = self.obj.price
                # 保持する株価を更新（順番依存）
                self.obj.price = price
                return self.obj
            else:
                # 株価に差が無ければ trend = 0 を維持
                return self.obj
        else:
            # trend が 0 でない時
            if self.cmp_psar(price):
                # _/_/_/_/_/_/
                # トレンド反転
                # _/_/_/_/_/_/
                self.obj.price = price
                # トレンド反転
                self.obj.trend *= -1
                # PSAR はこれまでの EP に更新（順番依存）
                self.obj.psar = self.obj.ep
                # EP は現在価格へ更新
                self.obj.ep = price
                # AF は初期値
                self.obj.af = self.af_init

                return self.obj
            else:
                if self.cmp_ep(price):
                    self.update_ep_af(price)

                self.obj.psar = self.obj.psar + self.obj.af * (self.obj.ep - self.obj.psar)
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
        """
        EP（極値）と AF（加速因数）の更新
        """
        # EP の更新
        self.obj.ep = price

        # AF の更新
        if self.obj.af < self.af_max - self.af_step:
            self.obj.af += self.af_step
