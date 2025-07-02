from collections import deque  # deque をインポート


class PSARObject:
    def __init__(self):
        self.price: float = 0.
        self.trend: int = 0
        self.ep: float = 0.
        self.af: float = -1.  # AF は 0 以上の実数
        self.psar: float = 0.
        self.epupd: int = 0  # EP が更新された回数
        self.total: int = 0  # 同一 Trend のカウンタ


class RealtimePSAR:
    def __init__(
            self,
            af_init: float = 0.00002,
            af_step: float = 0.00002,
            af_max: float = 0.002,
            rolling_n: int = 30
    ):
        self.af_init = af_init
        self.af_step = af_step
        self.af_max = af_max

        # PSARObject のインスタンス
        self.obj = PSARObject()

        # 最初のエントリは多数決ロジックで決定する
        self.rolling_n = rolling_n  # 固定長のデータ点数（n 個でローリング）
        self.prices_deque = deque(maxlen=self.rolling_n)  # deque を使用し、最大長を n に設定
        self.threshold_ratio = 2 / 3  # 多数決の閾値 (2:1 = 約66.6%)

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
                # トレンド反転
                self.obj.price = price
                self.obj.trend *= -1
                self.obj.psar = self.obj.ep
                self.obj.ep = price
                self.obj.af = self.af_init
                self.obj.epupd = 0  # EP 更新回数リセット
                self.obj.total = 0  # 同一 Trend のカウンタリセット
                return self.obj
            else:
                # トレンド維持
                if self.cmp_ep(price):
                    self.update_ep_af(price)
                self.obj.psar = self.obj.psar + self.obj.af * (self.obj.ep - self.obj.psar)
                self.obj.price = price
                self.obj.total += 1  # 同一 Trend のカウンタのインクリメント
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

    def decide_first_trend(self, price: float):
        """
        現時点の価格を基準に多数決を行い、固定長 n のデータで判定するメソッド。
        「低いデータが多い場合は上昇トレンド、そうでない場合は下降トレンド」として実装。
        """
        self.prices_deque.append(price)  # deque に追加 (maxlen により古いデータは自動削除)

        # 必要なデータ点数に達しているか確認 (deque の長さが n に達しているか)
        if len(self.prices_deque) < self.rolling_n:
            self.obj.price = price
            return self.obj

        # --- トレンド判定ロジック ---
        votes_lower = 0
        votes_higher = 0

        # deque 内の全要素を対象とする
        # ただし、現在価格と同じ場合はカウントしない
        for p in self.prices_deque:
            if p < price:  # 現時点での価格以下のデータ
                votes_lower += 1
            elif price < p:  # 現時点での価格より高いデータ
                votes_higher += 1

        total_votes = votes_lower + votes_higher

        if total_votes == 0:
            # 全要素が現在価格と同じ場合（現実には、ほぼありえない）
            self.obj.trend = 0
        elif self.threshold_ratio < votes_lower / total_votes:
            # 「低いデータが多い場合」は上昇トレンド
            self.obj.trend = +1
            self.obj.psar = min(self.prices_deque)  # PSARは期間内の最低値
        elif self.threshold_ratio < votes_higher / total_votes:
            # 「高いデータが多い場合」は下降トレンド
            self.obj.trend = -1
            self.obj.psar = max(self.prices_deque)  # PSARは期間内の最高値
        else:
            # 閾値を満たさない場合、トレンドは未決定 (n は固定なので増えない)
            self.obj.trend = 0

        if self.obj.trend != 0:
            # トレンドが決定された場合のみ、EPとAFを初期化
            self.obj.ep = price  # EPは現在価格から
            self.obj.af = self.af_init # AF の初期化
            self.obj.epupd = 0  # EP 更新回数のリセット
            self.obj.trend = 0  # 同一 Trend のカウンタのリセット
            # トレンド決定後、deque をクリア
            self.prices_deque.clear()

        self.obj.price = price
        return self.obj

    def update_ep_af(self, price: float):
        self.obj.ep = price
        self.obj.epupd += 1
        if self.obj.af < self.af_max - self.af_step:
            self.obj.af += self.af_step
