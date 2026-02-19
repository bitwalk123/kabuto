from collections import deque
from typing import Optional


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
