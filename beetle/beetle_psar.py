from collections import deque  # deque をインポート

from scipy.interpolate import make_smoothing_spline


class PSARObject:
    def __init__(self):
        self.af: float = -1.  # AF は 0 以上の実数
        self.distance: float = 0
        self.duration: int = 0
        self.ep: float = 0.
        self.epupd: int = 0
        self.price: float = 0.
        self.psar: float = 0.
        self.trend: int = 0
        self.ys: float = 0


class RealtimePSAR:
    def __init__(
            self,
            af_init: float = 0.000005,
            af_step: float = 0.000005,
            af_max: float = 0.005,
    ):
        self.af_init = af_init
        self.af_step = af_step
        self.af_max = af_max

        self.lam = 10 ** 7

        self.factor_d = 20  # 許容される ys と PSAR の最大差異

        # PSARObject のインスタンス
        self.obj = PSARObject()

        self.n_smooth_min = 60
        self.n_smooth_max = 600
        # 価格のみしか取得しないので、等間隔と仮定してカウンタとして使用する。
        # 【利点】ランチタイムのブランクを無視できる。
        self.t = 0.0
        self.t_deque = deque(maxlen=self.n_smooth_max)
        self.y_deque = deque(maxlen=self.n_smooth_max)

    def add(self, price: float) -> PSARObject:
        self.obj.price = price

        # Smoothing Spline
        self.t_deque.append(self.t)
        self.y_deque.append(price)
        self.t += 1.0
        if 5 < len(self.t_deque):
            spl = make_smoothing_spline(
                self.t_deque,
                self.y_deque,
                lam=self.lam
            )
            self.obj.ys = spl(self.t)
        else:
            self.obj.ys = price

        # Parabolic SAR
        if len(self.t_deque) < self.n_smooth_min:
            # -----------------------------------------------------------------
            # データ数が self.n_smooth_min 未満の場合は Parabolic SAR を適用しない。
            # -----------------------------------------------------------------
            # なにもしない
            pass
        elif self.obj.trend == 0:
            # -----------------------------------------------------------------
            # トレンドが 0 の時は寄り付き後で、ひとたび trend が +1 あるいは -1 になれば、
            # 以後はトレンド反転するので 0 になることは無い
            # -----------------------------------------------------------------
            # self.decide_first_trend(self.obj.ys)
            if self.y_deque[-2] < self.obj.ys:
                self.obj.trend = +1
                self.init_first_param(self.obj.ys)  # トレンド決定後の最初のパラメータ処理
            elif self.obj.ys < self.y_deque[-2]:
                self.obj.trend = -1
                self.init_first_param(self.obj.ys)  # トレンド決定後の最初のパラメータ処理
            else:
                # 大小を付けられなければ何もしない。
                pass
        elif self.cmp_psar(self.obj.ys):
            # -----------------------------------------------------------------
            # トレンド反転
            # -----------------------------------------------------------------
            self.obj.trend *= -1
            self.obj.psar = self.obj.ep
            self.obj.ep = self.obj.ys
            self.obj.af = self.af_init
            self.obj.epupd = 0
            self.obj.duration = 0
            # トレンド反転後の ys と psar の差異
            # これより差異が大きくなればトレンドをフォローするために使用（未実装）
            self.obj.distance = abs(self.obj.ys - self.obj.psar)
        else:
            # -----------------------------------------------------------------
            # トレンド維持
            # -----------------------------------------------------------------
            # EP更新かどうか判定
            if self.cmp_ep(self.obj.ys):
                # EP と AF の更新
                self.update_ep_af(self.obj.ys)
            # PSAR の更新
            self.obj.psar = self.obj.psar + self.obj.af * (self.obj.ep - self.obj.psar)

            # 許容される ys と PSAR の最大差異チェック
            if self.factor_d < abs(self.obj.psar - self.obj.ys):
                if 0 < self.obj.trend:
                    self.obj.psar = self.obj.ys - self.factor_d
                elif self.obj.trend < 0:
                    self.obj.psar = self.obj.ys + self.factor_d

            self.obj.duration += 1

        return self.obj

    def cmp_ep(self, y: float) -> bool:
        """
        EP更新か判定
        :param y:
        :return:
        """
        if 0 < self.obj.trend:
            if self.obj.ep < y:
                return True
            else:
                return False
        else:
            if y < self.obj.ep:
                return True
            else:
                return False

    def cmp_psar(self, y: float) -> bool:
        """
        トレンド反転か判定
        :param y:
        :return:
        """
        if 0 < self.obj.trend:
            if y < self.obj.psar:
                return True
            else:
                return False
        else:
            if self.obj.psar < y:
                return True
            else:
                return False

    def init_first_param(self, y):
        """
        トレンド決定後の最初のパラメータ処理
        :param y:
        :return:
        """
        self.obj.ep = y
        self.obj.af = self.af_init
        self.obj.psar = self.y_deque[-2]

    def update_ep_af(self, y: float):
        """
        EPとAFの更新
        :param y:
        :return:
        """
        self.obj.ep = y
        self.obj.epupd += 1
        self.obj.duration = 0
        if self.obj.af < self.af_max - self.af_step:
            self.obj.af += self.af_step
