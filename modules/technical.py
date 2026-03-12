from collections import deque
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


class RSI:
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.rsi = 50.0
        self.prev_rsi = 50.0
        self.prev_value = None
        self.avg_gain = None  # 初期化されていないことを明示
        self.avg_loss = None

        # 初期SMA計算用（最初のwindow_size期間のみ使用）
        self.init_gains = deque()
        self.init_losses = deque()

    def clear(self) -> None:
        self.rsi = 50.0
        self.prev_rsi = 50.0
        self.prev_value = None
        self.avg_gain = None
        self.avg_loss = None
        self.init_gains.clear()
        self.init_losses.clear()

    def getValue(self) -> float:
        return self.rsi

    def getSlope(self) -> float:
        return self.rsi - self.prev_rsi

    def update(self, value: float) -> float:
        if self.prev_value is None:
            self.prev_value = value
            return self.rsi

        # 価格変化を計算
        change = value - self.prev_value
        gain = change if change > 0.0 else 0.0
        loss = -change if change < 0.0 else 0.0

        # 平均が初期化されていない場合（最初のwindow_size期間）
        if self.avg_gain is None:
            self.init_gains.append(gain)
            self.init_losses.append(loss)

            # window_size個のデータが揃ったら初期平均を計算
            if len(self.init_gains) == self.window_size:
                self.avg_gain = sum(self.init_gains) / self.window_size
                self.avg_loss = sum(self.init_losses) / self.window_size
                # 初期データは不要になるのでクリア（メモリ節約）
                self.init_gains.clear()
                self.init_losses.clear()
        else:
            # Wilder's Smoothing
            self.avg_gain = (self.avg_gain * (self.window_size - 1) + gain) / self.window_size
            self.avg_loss = (self.avg_loss * (self.window_size - 1) + loss) / self.window_size

        # RSIを計算
        self.prev_rsi = self.rsi

        if self.avg_gain is not None:
            if self.avg_loss == 0.0:
                self.rsi = 100.0 if self.avg_gain > 0.0 else 50.0
            else:
                rs = self.avg_gain / self.avg_loss
                self.rsi = 100.0 - (100.0 / (1.0 + rs))

        self.prev_value = value
        return self.rsi
