import numpy as np


class PSARObject:
    def __init__(self):
        self.price: float = 0.
        self.trend: int = 0
        self.ep: float = 0.
        self.af: float = -1.  # AF は 0 以上の実数
        self.psar: float = 0.
        self.epupd: int = 0


class RealtimePSAR:
    def __init__(self, af_init=0.0, af_step=0.00002, af_max=0.002, initial_min_data_points=30):  # n の値を指定
        self.af_init = af_init
        self.af_step = af_step
        self.af_max = af_max
        self.initial_min_data_points = initial_min_data_points  # 初期トレンド決定に必要な最小データ点数

        self.obj = PSARObject()
        self.initial_prices = []  # 初期データ蓄積用のリストを追加

    def add(self, price: float) -> PSARObject:
        if self.obj.trend == 0:
            # 最初の add 呼び出しで obj.price を初期化し、同時に initial_prices にも追加
            if self.obj.price == 0 and not self.initial_prices:
                self.obj.price = price
                self.initial_prices.append(price)
                return self.obj
            else:
                return self.decide_first_trend(price)  # price は現在の価格
        else:
            # trend が 0 でない時 (元のコードのまま、変更なし)
            if self.cmp_psar(price):
                self.obj.price = price
                self.obj.trend *= -1
                self.obj.psar = self.obj.ep
                self.obj.ep = price
                self.obj.af = self.af_init
                self.obj.epupd = 0
                return self.obj
            else:
                if self.cmp_ep(price):
                    self.update_ep_af(price)
                self.obj.psar = self.obj.psar + self.obj.af * (self.obj.ep - self.obj.psar)
                self.obj.price = price
                return self.obj

    def decide_first_trend(self, price):
        """
        重み付けなしの多数決ロジックで初期トレンドを決定するメソッド。
        多数決条件の閾値を緩和。
        """
        self.initial_prices.append(price)  # 現在の価格を蓄積リストに追加

        if len(self.initial_prices) < self.initial_min_data_points:
            # 必要なデータ点数に達するまでトレンドは決定しない
            self.obj.price = price  # 最新価格は更新しておく
            return self.obj
        else:
            # データ点数が揃ったら、重み付けなし多数決で初期トレンドを決定
            up_votes = 0
            down_votes = 0
            total_votes = 0

            for i in range(1, self.initial_min_data_points):
                prev_price = self.initial_prices[i - 1]
                current_price = self.initial_prices[i]

                if current_price > prev_price:
                    up_votes += 1
                    total_votes += 1
                elif current_price < prev_price:
                    down_votes += 1
                    total_votes += 1

            # トレンドの最終決定
            # 閾値を 0.6 (60%以上) に変更
            threshold = 0.6

            if total_votes == 0:
                self.obj.trend = 0
            elif up_votes / total_votes >= threshold:  # 上昇が閾値以上
                self.obj.trend = +1
                self.obj.ep = max(self.initial_prices)
                self.obj.psar = min(self.initial_prices)
            elif down_votes / total_votes >= threshold:  # 下降が閾値以上
                self.obj.trend = -1
                self.obj.ep = min(self.initial_prices)
                self.obj.psar = max(self.initial_prices)
            else:
                self.obj.trend = 0

            if self.obj.trend != 0:
                self.obj.af = self.af_init
                self.obj.epupd = 0
                self.initial_prices = []

            self.obj.price = price
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
        if self.obj.af < self.af_max - self.af_step:
            self.obj.af += self.af_step