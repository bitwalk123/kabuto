from collections import deque  # deque をインポート

# from scipy.differentiate import derivative
from scipy.interpolate import make_smoothing_spline

from structs.app_enum import FollowType


class PSARObject:
    def __init__(self):
        self.af: float = -1.  # AF は 0 以上の実数
        self.distance: float = 0
        self.duration: int = 0
        self.ep: float = 0.
        self.epupd: int = 0
        self.overdrive = False
        self.price: float = 0.
        self.psar: float = 0.
        self.trend: int = 0
        self.y_sar: float = 0  # トレンド反転した時の価格
        self.ys: float = 0
        self.follow: FollowType = FollowType.PARABOLIC  # フォロータイプ
        # self.dys: float = 0 # 微係数


class RealtimePSAR:
    def __init__(self, dict_psar: dict):
        """
        リアルタイム用 Parabolic SAR
        :param dict_psar:
        """
        self.af_init: float | None = None
        self.af_step: float | None = None
        self.af_max: float | None = None
        self.factor_d: float | None = None
        self.factor_c: float | None = None
        self.lam: float | None = None
        self.n_smooth_min: int | None = None
        self.n_smooth_max: int | None = None
        self.t_deque: deque | None = None
        self.y_deque: deque | None = None

        # パラメータの設定
        self.setPSARParams(dict_psar)

        # 時刻用カウンター（実際の時刻を使わずに連続した数列を時刻代わりに利用）
        self.t: float = 0.0

        # PSARObject のインスタンス
        self.obj = PSARObject()

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
            # スムージング値
            self.obj.ys = spl(self.t)
            # 微係数の絶対値を算出
            # deriv = derivative(spl, self.t)
            # self.obj.dys = abs(deriv.df)
        else:
            self.obj.ys = price
            # self.obj.dys = 0

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
            self.obj.y_sar = price  # トレンド反転時の株価を保持
            # トレンド反転後の ys と psar の差異
            # これより差異が大きくなればトレンドをフォローするために使用（未実装）
            self.obj.distance = abs(self.obj.ys - self.obj.psar)
            self.obj.follow = FollowType.PARABOLIC  # デフォルトのフォロータイプ
        else:
            # -----------------------------------------------------------------
            # トレンド維持
            # -----------------------------------------------------------------
            # EP更新かどうか判定
            if self.cmp_ep(self.obj.ys):
                self.update_ep_af(self.obj.ys)  # EP と AF の更新

            # 許容される ys と PSAR の最大差異チェック
            delta_psar = abs(self.obj.psar - self.obj.ys)
            if self.factor_d < delta_psar:
                # ひとたび OVERDRIVE モードになれば
                # 現在のところトレンド反転するまでこのモードを続ける
                self.obj.follow = FollowType.OVERDRIVE
                self.obj.psar = self.obj.ys - self.factor_d * self.obj.trend
            elif self.obj.follow == FollowType.OVERDRIVE:
                self.obj.psar = self.obj.ys - delta_psar * self.factor_c * self.obj.trend
            else:
                # Parabolic SAR の更新
                self.obj.psar = self.obj.psar + self.obj.af * (self.obj.ep - self.obj.psar)

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

    def setOverDriveStatus(self, state: bool):
        if state:
            self.obj.follow = FollowType.OVERDRIVE
        else:
            self.obj.follow = FollowType.PARABOLIC

    def setPSARParams(self, dict_psar):
        """
        パラメータの設定
        :param dict_psar:
        :return:
        """
        # for Parabolic SAR
        self.af_init = dict_psar["af_init"]
        self.af_step = dict_psar["af_step"]
        self.af_max = dict_psar["af_max"]
        self.factor_d = dict_psar["factor_d"]  # 許容される ys と PSAR の最大差異 (delta)
        self.factor_c = dict_psar["factor_c"]  # トレンド追跡 (chase) ファクター
        # for smoothing
        self.lam = 10. ** dict_psar["power_lam"]
        self.n_smooth_min = dict_psar["n_smooth_min"]
        self.n_smooth_max = dict_psar["n_smooth_max"]
        # スムージングの為に保持するデータ列
        # 価格のみしか取得しないので、等間隔と仮定してカウンタとして使用する。
        # 【利点】ランチタイムのブランクを無視できる。
        self.t_deque = deque(maxlen=self.n_smooth_max)
        self.y_deque = deque(maxlen=self.n_smooth_max)

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
