import time
from collections import deque
import numpy as np
from talib import stream


# 現在の実装
class MovingAverage:
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.queue = deque()
        self.running_sum = 0.0
        self.ma = 0.0
        self.prev_ma = 0.0

    def update(self, value: float) -> float:
        if len(self.queue) >= self.window_size:
            self.running_sum -= self.queue.popleft()
        self.queue.append(value)
        self.running_sum += value
        self.prev_ma = self.ma
        self.ma = self.running_sum / len(self.queue)
        return self.ma


# TA-Lib版
class MovingAverageTA:
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
        self.ma = 0.0
        self.prev_ma = 0.0

    def update(self, value: float) -> float:
        self.buffer.append(value)
        if len(self.buffer) >= self.window_size:
            self.prev_ma = self.ma
            buffer_array = np.array(self.buffer, dtype=float)
            self.ma = stream.SMA(buffer_array, timeperiod=self.window_size)
        return self.ma


# ベンチマーク
def benchmark(ma_class, iterations=100000):
    ma = ma_class(30)
    start = time.perf_counter()
    for i in range(iterations):
        ma.update(float(i))
    elapsed = time.perf_counter() - start
    return elapsed


# 実行
time_custom = benchmark(MovingAverage)
time_talib = benchmark(MovingAverageTA)

print(f"現在の実装: {time_custom:.4f}秒")
print(f"TA-Lib版:   {time_talib:.4f}秒")
print(f"速度比:     {time_talib / time_custom:.2f}倍遅い")