import numpy as np


class PSARObject:
    def __init__(self):
        self.price: float = 0.
        self.trend: int = 0  # 0:未確定, +1:上昇, -1:下降
        self.ep: float = 0.
        self.af: float = -1.
        self.psar: float = 0.
        self.epupd: int = 0


class RealtimePSAR:
    def __init__(self, af_init=0.0, af_step=0.00002, af_max=0.002, initial_ma_period=60):  # n を initial_ma_period に変更
        self.af_init = af_init
        self.af_step = af_step
        self.af_max = af_max
        self.initial_ma_period = initial_ma_period  # 移動平均線の計算期間 (n の値)
        self.initial_prices = []  # 寄り付き後の初期データを蓄積
        # self.logger = logging.getLogger(__name__)

        self.obj = PSARObject()

    def add(self, price: float) -> PSARObject:
        if self.obj.trend == 0:  # 初期トレンドがまだ確定していない場合
            self.initial_prices.append(price)

            # 移動平均線の計算に必要なデータ点数に達している場合のみ、トレンド決定ロジックを試行
            if len(self.initial_prices) >= self.initial_ma_period:
                self._determine_initial_trend_and_ep_moving_average()  # メソッド名を変更

                if self.obj.trend != 0:  # トレンドが決定できた場合のみ、初期化して次へ進む
                    # logger.info(f"Initial PSAR trend determined: {self.obj.trend}, EP: {self.obj.ep}, PSAR: {self.obj.psar}")
                    self.initial_prices = []  # 初期データはもう不要なのでクリア
                    self.obj.price = price  # 最新価格をセット
                    # ここで return せずに、下の通常のPSAR計算ロジックに進む
                else:
                    # トレンドがまだ決定できていない場合 (移動平均線条件を満たさない)
                    self.obj.price = price  # 最新価格は更新しておく
                    return self.obj  # トレンド未確定なので、PSARは更新しない
            else:
                # 最低点数に達していない場合
                self.obj.price = price  # 最新価格は更新しておく
                return self.obj  # トレンド未確定なので、PSARは更新しない

        # ここから既存のPSAR計算ロジック
        if self.obj.trend != 0:  # トレンドが確立している場合
            # PSARが現在価格を越えないように調整 (通常のPSARロジックの必須部分)
            if self.cmp_psar(price):  # トレンド反転
                # logger.info("Trend reversal detected!")
                self.obj.price = price
                self.obj.trend *= -1
                self.obj.psar = self.obj.ep
                self.obj.ep = price
                self.obj.af = self.af_init
                self.obj.epupd = 0
            else:  # トレンド継続
                if self.cmp_ep(price):
                    self.update_ep_af(price)

                if self.obj.trend == +1:  # 上昇トレンド
                    self.obj.psar = max(self.obj.psar + self.obj.af * (self.obj.ep - self.obj.psar), price)
                else:  # 下降トレンド
                    self.obj.psar = min(self.obj.psar + self.obj.af * (self.obj.ep - self.obj.psar), price)

            self.obj.price = price
            return self.obj
        else:
            self.obj.price = price
            return self.obj

    def _calculate_sma(self, data, period):
        """指定された期間の単純移動平均を計算する"""
        if len(data) < period:
            return None
        return sum(data[-period:]) / period

    def _determine_initial_trend_and_ep_moving_average(self):
        """
        蓄積されたデータと移動平均線を使ってトレンドを決定する。
        """
        prices_len = len(self.initial_prices)
        if prices_len < self.initial_ma_period:
            self.obj.trend = 0
            return

            # 最新の移動平均を計算
        current_ma = self._calculate_sma(self.initial_prices, self.initial_ma_period)

        if current_ma is None:  # データが足りない場合はトレンド決定せず
            self.obj.trend = 0
            return

        latest_price = self.initial_prices[-1]

        # トレンドの判定基準
        # 最新価格が移動平均線を上回っていれば上昇トレンド
        if latest_price > current_ma:
            self.obj.trend = +1  # 上昇トレンドと決定
        # 最新価格が移動平均線を下回っていれば下降トレンド
        elif latest_price < current_ma:
            self.obj.trend = -1  # 下降トレンドと決定
        else:
            # 最新価格と移動平均線が同じ場合や、明確な方向性がない場合はトレンド未決定 (trend=0 のまま)
            self.obj.trend = 0
            return

            # トレンドが決定できた場合のみ、EPとPSARを初期設定
        if self.obj.trend == +1:  # 上昇トレンド
            self.obj.ep = min(self.initial_prices[-self.initial_ma_period:])  # MA期間内の最安値をEPに
            self.obj.psar = max(self.initial_prices[-self.initial_ma_period:])  # PSARの初期値はMA期間内の最高値
        else:  # self.obj.trend == -1 (下降トレンド)
            self.obj.ep = max(self.initial_prices[-self.initial_ma_period:])  # MA期間内の最高値をEPに
            self.obj.psar = min(self.initial_prices[-self.initial_ma_period:])  # PSARの初期値はMA期間内の最安値

        self.obj.af = self.af_init
        self.obj.epupd = 0
        # self.obj.price は add メソッドで既に最新価格になっている（呼び出し元で設定される）

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