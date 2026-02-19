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
