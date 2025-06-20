import numpy as np

class PSARObject:
    def __init__(self):
        self.price: float = 0.
        self.trend: int = 0 # 0:未確定, +1:上昇, -1:下降
        self.ep: float = 0.
        self.af: float = -1.
        self.psar: float = 0.
        self.epupd: int = 0

class RealtimePSAR:
    def __init__(self, af_init=0.0, af_step=0.00002, af_max=0.002, initial_min_data_points=30):
        self.af_init = af_init
        self.af_step = af_step
        self.af_max = af_max
        self.initial_min_data_points = initial_min_data_points # n の値 (最低点数)
        self.initial_prices = [] # 寄り付き後の初期データを蓄積
        # self.logger = logging.getLogger(__name__) # ロギングは既存の環境に合わせて追加

        self.obj = PSARObject()

    def add(self, price: float) -> PSARObject:
        if self.obj.trend == 0: # 初期トレンドがまだ確定していない場合
            self.initial_prices.append(price)

            # 最低点数に達している場合のみ、トレンド決定ロジックを試行
            if len(self.initial_prices) >= self.initial_min_data_points:
                self._determine_initial_trend_and_ep_weighted_diff()

                if self.obj.trend != 0: # トレンドが決定できた場合のみ、初期化して次へ進む
                    # logger.info(f"Initial PSAR trend determined: {self.obj.trend}, EP: {self.obj.ep}, PSAR: {self.obj.psar}")
                    self.initial_prices = [] # 初期データはもう不要なのでクリア
                    self.obj.price = price # 最新価格をセット
                    # ここで return せずに、下の通常のPSAR計算ロジックに進む
                else:
                    # トレンドがまだ決定できていない場合 (多数決条件を満たさない)
                    self.obj.price = price # 最新価格は更新しておく
                    return self.obj # トレンド未確定なので、PSARは更新しない
            else:
                # 最低点数に達していない場合
                self.obj.price = price # 最新価格は更新しておく
                return self.obj # トレンド未確定なので、PSARは更新しない

        # ここから既存のPSAR計算ロジック
        # self.obj.trend が 0 でない場合、または上記で 0 から変更された場合
        if self.obj.trend != 0: # トレンドが確立している場合
            # PSARが現在価格を越えないように調整 (通常のPSARロジックの必須部分)
            if self.cmp_psar(price): # トレンド反転
                # logger.info("Trend reversal detected!") # 必要に応じてロギング
                self.obj.price = price
                self.obj.trend *= -1
                self.obj.psar = self.obj.ep
                self.obj.ep = price
                self.obj.af = self.af_init
                self.obj.epupd = 0
            else: # トレンド継続
                if self.cmp_ep(price):
                    self.update_ep_af(price)

                if self.obj.trend == +1: # 上昇トレンド
                    # PSARが現在の価格より高くなるのを防ぐためmax(psar, price)
                    self.obj.psar = max(self.obj.psar + self.obj.af * (self.obj.ep - self.obj.psar), price)
                else: # 下降トレンド
                    # PSARが現在の価格より低くなるのを防ぐためmin(psar, price)
                    self.obj.psar = min(self.obj.psar + self.obj.af * (self.obj.ep - self.obj.psar), price)

            self.obj.price = price # 最新価格を保持
            return self.obj
        else:
            self.obj.price = price
            return self.obj


    def _determine_initial_trend_and_ep_weighted_diff(self):
        """
        蓄積されたデータを使って、隣接する2点間の差分に重み付けをして多数決によりトレンドを決定する。
        2:1（2/3以上）多数決条件を満たさない限り、トレンドは決定しない (trend=0)。
        """
        prices_len = len(self.initial_prices)
        if prices_len < 2:
            self.obj.trend = 0
            return

        weighted_up_sum = 0.0
        weighted_down_sum = 0.0

        for i in range(prices_len - 1):
            diff = self.initial_prices[i+1] - self.initial_prices[i]
            weight = i + 1

            if diff > 0:
                weighted_up_sum += weight
            elif diff < 0:
                weighted_down_sum -= weight

        total_weighted_absolute_votes = abs(weighted_up_sum) + abs(weighted_down_sum)

        if total_weighted_absolute_votes == 0:
            self.obj.trend = 0
            return

        # ★★★ ここを修正: 多数決の基準を 2:1 (2/3 以上) に変更 ★★★
        required_weighted_votes_for_majority = total_weighted_absolute_votes * (2/3)

        if weighted_up_sum > required_weighted_votes_for_majority:
            self.obj.trend = +1 # 上昇トレンドと決定
        elif abs(weighted_down_sum) > required_weighted_votes_for_majority:
            self.obj.trend = -1 # 下降トレンドと決定
        else:
            # 2:1以上の多数決条件を満たさない場合はトレンド未決定 (trend=0 のまま)
            self.obj.trend = 0
            return

        # EPとPSARの初期値設定（これは前回の高橋さんのご指示通り）
        if self.obj.trend == +1: # 上昇トレンド
            self.obj.ep = min(self.initial_prices)
            self.obj.psar = self.initial_prices[0] # PSARの初期値は最初の価格
        else: # self.obj.trend == -1 (下降トレンド)
            self.obj.ep = max(self.initial_prices)
            self.obj.psar = self.initial_prices[0] # PSARの初期値は最初の価格

        print(weighted_up_sum, required_weighted_votes_for_majority, self.obj.ep, self.obj.psar)

        self.obj.af = self.af_init
        self.obj.epupd = 0

    def cmp_ep(self, price: float) -> bool:
        # ... (既存のコードと同じ)
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
        # ... (既存のコードと同じ)
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
        # ... (既存のコードと同じ)
        self.obj.ep = price
        self.obj.epupd += 1

        if self.obj.af < self.af_max - self.af_step:
            self.obj.af += self.af_step