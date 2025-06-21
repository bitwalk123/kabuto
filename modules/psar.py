import numpy as np
from collections import deque  # deque をインポート


class PSARObject:
    def __init__(self):
        self.price: float = 0.
        self.trend: int = 0
        self.ep: float = 0.
        self.af: float = -1.  # AF は 0 以上の実数
        self.psar: float = 0.
        self.epupd: int = 0


class RealtimePSAR:
    # initial_min_data_points は n を固定長として使うため、ここでは n と解釈
    def __init__(self, af_init=0.0, af_step=0.00002, af_max=0.002, n=30):
        self.af_init = af_init
        self.af_step = af_step
        self.af_max = af_max
        self.n = n  # 固定長のデータ点数として n を使用

        self.obj = PSARObject()
        # deque を使用し、最大長を n に設定
        self.prices_deque = deque(maxlen=self.n)

    def add(self, price: float) -> PSARObject:
        if self.obj.trend == 0:
            # 最初の add 呼び出しで obj.price を初期化し、同時に prices_deque にも追加
            if self.obj.price == 0 and not self.prices_deque:
                self.obj.price = price
                self.prices_deque.append(price)
                return self.obj
            else:
                return self.decide_first_trend(price)
        else:
            # trend が 0 でない時 (既存のロジック、変更なし)
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
        現時点の価格を基準に多数決を行い、固定長 n のデータで判定するメソッド。
        「低いデータが多い場合は上昇トレンド、そうでない場合は下降トレンド」として実装。
        """
        self.prices_deque.append(price)  # deque に追加 (maxlen により古いデータは自動削除)

        # 必要なデータ点数に達しているか確認 (deque の長さが n に達しているか)
        if len(self.prices_deque) < self.n:
            self.obj.price = price
            return self.obj

        # --- トレンド判定ロジック ---
        lower_votes = 0
        higher_votes = 0

        # deque 内の全要素を対象とする
        for p in self.prices_deque:
            if p <= price:  # 現時点での価格以下のデータ
                lower_votes += 1
            else:  # 現時点での価格より高いデータ
                higher_votes += 1

        total_votes = lower_votes + higher_votes

        # 多数決の閾値 (2:1 = 約66.6%)
        threshold_ratio = 2 / 3  # 今回のご要望に合わせて 2/3 に戻します

        if total_votes == 0:
            self.obj.trend = 0
        elif lower_votes / total_votes >= threshold_ratio:
            # 「低いデータが多い場合」は上昇トレンド
            self.obj.trend = +1
            self.obj.ep = max(self.prices_deque)  # EPは期間内の最高値
            self.obj.psar = min(self.prices_deque)  # PSARは期間内の最低値

            # トレンド決定後、deque をクリアして次のトレンド決定に備える（必要であれば）
            # もしトレンドが一度決定されたら、dequeはもう使用しないためクリア
            self.prices_deque.clear()

        elif higher_votes / total_votes >= threshold_ratio:
            # 「高いデータが多い場合」は下降トレンド
            self.obj.trend = -1
            self.obj.ep = min(self.prices_deque)  # EPは期間内の最低値
            self.obj.psar = max(self.prices_deque)  # PSARは期間内の最高値

            # トレンド決定後、deque をクリア
            self.prices_deque.clear()
        else:
            # 閾値を満たさない場合、トレンドは未決定 (n は固定なので増えない)
            self.obj.trend = 0

            # トレンドが決定された場合のみ、AFを初期化
        if self.obj.trend != 0:
            self.obj.af = self.af_init
            self.obj.epupd = 0

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