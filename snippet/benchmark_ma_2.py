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
# window_sizeを変えて測定
for window_size in [10, 30, 100, 300]:
    print(f"\n=== window_size = {window_size} ===")

    ma1 = MovingAverage(window_size)
    start = time.perf_counter()
    for i in range(100000):
        ma1.update(float(i))
    time1 = time.perf_counter() - start

    ma2 = MovingAverageTA(window_size)
    start = time.perf_counter()
    for i in range(100000):
        ma2.update(float(i))
    time2 = time.perf_counter() - start

    print(f"現在の実装: {time1:.4f}秒")
    print(f"TA-Lib版:   {time2:.4f}秒")
    print(f"速度比:     {time2 / time1:.2f}倍")