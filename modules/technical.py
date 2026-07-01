import math
from collections import deque
from math import sqrt
from typing import Optional

from sortedcontainers import SortedList


class MovingAverage:
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.queue = deque()
        self.running_sum = 0.0
        self.ma = 0.0
        self.prev_ma = 0.0  # 直前の MA を保持

    def clear(self) -> None:
        self.queue.clear()
        self.running_sum = 0.0
        self.ma = 0.0
        self.prev_ma = 0.0

    def getValue(self) -> float:
        return self.ma

    def getSlope(self) -> float:
        # s = ma_current - ma_prev
        return self.ma - self.prev_ma

    def update(self, value: float) -> float:
        # 古い値を取り除く
        if len(self.queue) >= self.window_size:
            self.running_sum -= self.queue.popleft()

        # 新しい値を追加
        self.queue.append(value)
        self.running_sum += value

        # MA を更新（更新前に prev_ma を保存）
        self.prev_ma = self.ma
        self.ma = self.running_sum / len(self.queue)

        return self.ma


class MovingIQR:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.data = SortedList()
        self.queue: deque[float] = deque()

        self.q1: Optional[float] = None
        self.q3: Optional[float] = None
        self.iqr: Optional[float] = None

    def clear(self) -> None:
        self.data.clear()
        self.queue.clear()
        self.q1 = None
        self.q3 = None
        self.iqr = None

    def update(self, value: float) -> tuple[float | None, float | None, float | None]:
        # 新しい値を追加
        self.data.add(value)
        self.queue.append(value)

        # window_size を超えたら古い値を削除
        if self.window_size < len(self.data):
            old = self.queue.popleft()
            self.data.remove(old)

        n = len(self.data)
        if n == 0:
            return None, None, None

        # Q1, Q3 のインデックス（window_size=100 なら 25, 75）
        idx_q1 = int(n * 0.25)
        idx_q3 = int(n * 0.75)

        self.q1 = self.data[idx_q1]
        self.q3 = self.data[idx_q3]
        self.iqr = self.q3 - self.q1

        return self.q1, self.q3, self.iqr

    def getValue(self) -> tuple[float | None, float | None, float | None]:
        if self.q1 is None:
            return None, None, None
        return self.q1, self.q3, self.iqr

    def getLower(self) -> float | None:
        # データがまだ無い
        if self.q1 is None:
            return None

        # データが1つだけ → その値を返す
        if self.iqr is None:
            return self.q1

        return self.q1 - 1.5 * self.iqr

    def getUpper(self) -> float | None:
        if self.q3 is None:
            return None

        if self.iqr is None:
            return self.q3

        return self.q3 + 1.5 * self.iqr


class VWAP:
    def __init__(self):
        self.running_pv: float = 0.0
        self.running_vol: float = 0.0
        self.vwap: float = 0.0
        self.prev_vwap: float = 0.0
        self.prev_volume: Optional[float] = None

    def clear(self) -> None:
        self.running_pv = 0.0
        self.running_vol = 0.0
        self.vwap = 0.0
        self.prev_vwap = 0.0
        self.prev_volume = None

    def getValue(self) -> float:
        return self.vwap

    def getSlope(self) -> float:
        return self.vwap - self.prev_vwap

    def update(self, price: float, cumulative_volume: float) -> float:
        # 初回ティック：始値を VWAP として採用
        if self.prev_volume is None:
            self.prev_volume = cumulative_volume
            self.vwap = price
            self.prev_vwap = price
            return self.vwap

        # 出来高の増加量
        vol_delta = cumulative_volume - self.prev_volume
        self.prev_volume = cumulative_volume

        # 異常値（出来高減少）は無視
        if vol_delta <= 0:
            return self.vwap

        # 加重合計を更新
        self.running_pv += price * vol_delta
        self.running_vol += vol_delta

        # VWAP 更新
        self.prev_vwap = self.vwap
        self.vwap = self.running_pv / self.running_vol

        return self.vwap


class ROC:
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.queue = deque()
        self.roc = 0.0
        self.prev_roc = 0.0  # 直前の ROC を保持

    def clear(self) -> None:
        self.queue.clear()
        self.roc = 0.0
        self.prev_roc = 0.0

    def getValue(self) -> float:
        return self.roc

    def update(self, value: float) -> float:
        # 過去データを保持
        self.queue.append(value)

        # 必要以上に古いデータは削除
        if len(self.queue) > self.window_size + 1:
            self.queue.popleft()

        # 更新前の値を保存
        self.prev_roc = self.roc

        # 十分なデータが揃っていない場合
        if len(self.queue) <= self.window_size:
            self.roc = 0.0
            return self.roc

        past_value = self.queue[0]

        # ゼロ除算防止
        if past_value == 0:
            self.roc = 0.0
        else:
            self.roc = ((value - past_value) / past_value) * 100.0

        return self.roc


class RSI:
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.rsi = 0.5
        self.prev_rsi = 0.5
        self.value_prev = None
        self.avg_gain = None
        self.avg_loss = None
        # 初期化フェーズ用
        self.init_gain_sum = 0.0
        self.init_loss_sum = 0.0
        self.init_count = 0
        self.alpha = 1.0 / window_size
        self.one_minus_alpha = 1.0 - self.alpha

    def clear(self):
        self.__init__(self.window_size)

    def getValue(self) -> float:
        return self.rsi

    def getSlope(self) -> float:
        return self.rsi - self.prev_rsi

    def update(self, value: float) -> float:
        if self.value_prev is None:
            self.value_prev = value
            return self.rsi

        change = value - self.value_prev

        if change > 0.0:
            gain = change
            loss = 0.0
        elif change < 0.0:
            gain = 0.0
            loss = -change
        else:
            gain = 0.0
            loss = 0.0

        # 初期 SMA フェーズ
        if self.avg_gain is None:
            self.init_gain_sum += gain
            self.init_loss_sum += loss
            self.init_count += 1
            if self.init_count == self.window_size:
                self.avg_gain = self.init_gain_sum / self.window_size
                self.avg_loss = self.init_loss_sum / self.window_size
        else:
            # Wilder smoothing
            self.avg_gain = self.one_minus_alpha * self.avg_gain + self.alpha * gain
            self.avg_loss = self.one_minus_alpha * self.avg_loss + self.alpha * loss

        # RSI 計算（簡略版）
        self.prev_rsi = self.rsi
        if self.avg_gain is not None:
            total = self.avg_gain + self.avg_loss
            self.rsi = self.avg_gain / total if total > 0.0 else 0.5

        self.value_prev = value
        return self.rsi


class Momentum:
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.queue = deque()
        self.momentum = 0.0
        self.prev_momentum = 0.0  # 直前の Momentum を保持

    def clear(self) -> None:
        self.queue.clear()
        self.momentum = 0.0
        self.prev_momentum = 0.0

    def getValue(self) -> float:
        return self.momentum

    def getSlope(self) -> float:
        # モメンタムの変化率（加速度的な情報）
        return self.momentum - self.prev_momentum

    def update(self, value: float) -> float:
        # 新しい値を追加
        self.queue.append(value)

        """
        # キューがwindow_sizeに達するまでは0を返す
        if len(self.queue) < self.window_size:
            return self.momentum
        """

        # window_sizeを超えたら古い値を削除
        if len(self.queue) > self.window_size:
            self.queue.popleft()

        # Momentum を更新（更新前に prev_momentum を保存）
        self.prev_momentum = self.momentum

        # Momentum = 現在値 - window_size期間前の値
        self.momentum = self.queue[-1] - self.queue[0]

        return self.momentum


class PurePursuitFollower:
    def __init__(
            self,
            trend_period: int = 5,
            gain: float = 0.15,
            predict_gain: float = 0.5,
    ):
        self.trend_period = trend_period
        self.gain = gain
        self.predict_gain = predict_gain

        self.queue = deque()

        self.follower = 0.0
        self.momentum = 0.0
        # self.follower_prev = 0.0
        self.initialized = False

    def clear(self) -> None:
        self.queue.clear()

        self.follower = 0.0
        self.momentum = 0.0
        # self.follower_prev = 0.0
        self.initialized = False

    def getValue(self) -> tuple[float, float]:
        return self.follower, self.momentum

    def update(self, price: float) -> tuple[float, float]:
        #
        # 初回
        #
        if not self.initialized:
            self.follower = price
            # self.follower_prev = price
            self.initialized = True

        #
        # 価格履歴追加
        #
        self.queue.append(price)

        #
        # トレンド推定
        #
        if len(self.queue) > self.trend_period:
            delayed_price = self.queue[0]

            # trend_period本での値動き
            velocity = price - delayed_price

            # 先読み価格
            target = price + self.predict_gain * velocity

            self.queue.popleft()
        else:
            target = price

        #
        # Pure Pursuit 更新
        #
        # self.follower_prev = self.follower
        error = target - self.follower
        self.follower += self.gain * error
        # follower 更新前の追従誤差
        self.momentum = error
        # self.momentum = self.follower - self.follower_prev

        return self.follower, self.momentum


class WMA:
    """加重移動平均"""

    def __init__(self, window_size: int):
        self.window_size = max(1, window_size)
        self.queue = deque()

        self.value = 0.0
        self.prev_value = 0.0

        # 1,2,3,...,N
        self.weights = list(range(1, self.window_size + 1))
        self.weight_sum = sum(self.weights)

    def clear(self):
        self.queue.clear()
        self.value = 0.0
        self.prev_value = 0.0

    def getValue(self) -> float:
        return self.value

    def update(self, x: float) -> float:
        if len(self.queue) >= self.window_size:
            self.queue.popleft()

        self.queue.append(x)

        self.prev_value = self.value

        q = list(self.queue)

        # ウィンドウ未充足時も計算する
        weights = self.weights[-len(q):]
        self.value = sum(v * w for v, w in zip(q, weights)) / sum(weights)

        return self.value


class HMA:
    """
    Hull Moving Average
    MovingAverage と同じインターフェイス
    """

    def __init__(self, window_size: int):
        self.window_size = window_size

        half = max(1, window_size // 2)
        root = max(1, int(sqrt(window_size)))

        self.wma_half = WMA(half)
        self.wma_full = WMA(window_size)
        self.wma_final = WMA(root)

        self.ma = 0.0
        self.prev_ma = 0.0

    def clear(self) -> None:
        self.wma_half.clear()
        self.wma_full.clear()
        self.wma_final.clear()

        self.ma = 0.0
        self.prev_ma = 0.0

    def getValue(self) -> float:
        return self.ma

    def update(self, value: float) -> float:
        half_ma = self.wma_half.update(value)
        full_ma = self.wma_full.update(value)

        raw_hma = 2.0 * half_ma - full_ma

        self.prev_ma = self.ma
        self.ma = self.wma_final.update(raw_hma)

        return self.ma


class RunningStatistics:
    """
    Welford statistics の実装（Z-score のみ）
    """

    def __init__(self):
        self.count = 0

        # Welford の内部状態
        self.mean = 0.0
        self.M2 = 0.0

        # 統計量
        self.variance = 0.0
        self.std = 0.0

        # オシレータとして利用する値
        self.zscore = 0.0
        self.prev_zscore = 0.0

    def clear(self) -> None:
        self.count = 0

        # Welford の内部状態
        self.mean = 0.0
        self.M2 = 0.0

        # 統計量
        self.variance = 0.0
        self.std = 0.0

        # オシレータとして利用する値
        self.zscore = 0.0
        self.prev_zscore = 0.0

    def getValue(self) -> float:
        return self.zscore

    def getSlope(self) -> float:
        return self.zscore - self.prev_zscore

    def update(self, value: float) -> float:
        #
        # 1. 更新前の統計量で現在値を評価
        #
        self.prev_zscore = self.zscore

        if self.count > 1 and self.std > 0.0:
            self.zscore = (value - self.mean) / self.std
        else:
            self.zscore = 0.0

        #
        # 2. Welford Algorithm による統計量更新
        #
        self.count += 1

        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean

        self.M2 += delta * delta2

        if self.count > 1:
            self.variance = self.M2 / self.count
            self.std = math.sqrt(self.variance)
        else:
            self.variance = 0.0
            self.std = 0.0

        return self.zscore


class EfficiencyRatio:
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.queue = deque()

        self.er = 0.0
        self.prev_er = 0.0

    def clear(self) -> None:
        self.queue.clear()

        self.er = 0.0
        self.prev_er = 0.0

    def getValue(self) -> float:
        return self.er

    def getSlope(self) -> float:
        return self.er - self.prev_er

    def update(self, value: float) -> float:

        # 新しい価格を追加
        self.queue.append(value)

        # window_size を超えたら古い価格を削除
        if len(self.queue) > self.window_size:
            self.queue.popleft()

        self.prev_er = self.er

        # データ不足
        if len(self.queue) < 2:
            self.er = 0.0
            return self.er

        # Direction（始点と終点の距離）
        direction = abs(self.queue[-1] - self.queue[0])

        # Volatility（実際に歩いた距離）
        volatility = 0.0
        for i in range(1, len(self.queue)):
            volatility += abs(self.queue[i] - self.queue[i - 1])

        if volatility > 0.0:
            self.er = direction / volatility
        else:
            self.er = 0.0

        return self.er
