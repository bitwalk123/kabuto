from modules.technical import MovingAverage
from funcs.technical_backup import SimpleSlope


class RealTimeDMA:
    def __init__(self, period_1=60, period_2=540,
                 slope_window=5, slope_threshold=0.05,
                 angle_threshold=0.1):

        self.obj_ma1 = MovingAverage(window_size=period_1)
        self.obj_ma2 = MovingAverage(window_size=period_2)

        self.dma_prev = None

        # Slope 計算をクラス化
        self.slope1 = SimpleSlope(window=slope_window)
        self.slope2 = SimpleSlope(window=slope_window)

        self.slope_threshold = slope_threshold
        self.angle_threshold = angle_threshold

    def update(self, price):
        ma1 = self.obj_ma1.update(price)
        ma2 = self.obj_ma2.update(price)

        if ma1 is None or ma2 is None:
            return None

        dma = ma1 - ma2

        # クロス判定
        if self.dma_prev is None:
            cross = 0
        else:
            if self.dma_prev < 0 < dma:
                cross = +1
            elif dma < self.dma_prev < 0:
                cross = -1
            else:
                cross = 0
        self.dma_prev = dma

        # slope 計算（クラスに委譲）
        slope1 = self.slope1.update(ma1)
        slope2 = self.slope2.update(ma2)

        # 角度の強さ（atan 不要）
        slope_diff = abs(slope1 - slope2)
        strong_angle = slope_diff > self.angle_threshold

        strong_slope = abs(slope1) > self.slope_threshold

        entry = (cross != 0) and strong_slope and strong_angle

        return {
            "MA1": ma1,
            "MA2": ma2,
            "DMA": dma,
            "Cross": cross,
            "Slope_MA1": slope1,
            "Slope_MA2": slope2,
            "SlopeDiff": slope_diff,
            "StrongSlope": strong_slope,
            "StrongAngle": strong_angle,
            "Entry": entry,
        }
