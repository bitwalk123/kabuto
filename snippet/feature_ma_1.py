from collections import deque

from funcs.technical import MovingAverage


class RealTimeDMA:
    """
    RealTimeDMA:
    リアルタイムで MA1 / MA2 / DMA / クロス / 傾き を逐次更新する軽量クラス。
    FeatureProvider で使用し、ObservationManager に dict を渡す用途を想定。
    """

    def __init__(self, period_1=60, period_2=540, slope_window=5, threshold=0.05):
        self.obj_ma1 = MovingAverage(window_size=period_1)
        self.obj_ma2 = MovingAverage(window_size=period_2)
        # ---------------------------------------------------------------------
        self.dma_prev = None
        self.ma1_prev = None
        # ---------------------------------------------------------------------
        self.slope_buf = deque(maxlen=slope_window)
        self.threshold = threshold

    def update(self, price):
        ma1 = self.obj_ma1.update(price)
        ma2 = self.obj_ma2.update(price)
        # ---------------------------------------------------------------------
        # 初期化段階
        # ただし、MovingAverage は None を返さない仕様なので不要かも
        if ma1 is None or ma2 is None:
            return None
        # ---------------------------------------------------------------------
        # DMA, delta MA
        dma = ma1 - ma2
        # ---------------------------------------------------------------------
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
        # ---------------------------------------------------------------------
        # 傾き（diff → rolling mean）
        if self.ma1_prev is None:
            diff_ma1 = 0.0
        else:
            diff_ma1 = ma1 - self.ma1_prev
        self.ma1_prev = ma1
        self.slope_buf.append(diff_ma1)
        slope_ma1 = sum(self.slope_buf) / len(self.slope_buf)
        # ---------------------------------------------------------------------
        # 強い傾き
        strong_slope = abs(slope_ma1) > self.threshold
        # ---------------------------------------------------------------------
        # エントリー条件
        entry = (cross != 0) and strong_slope
        # ---------------------------------------------------------------------
        return {
            "MA1": ma1,
            "MA2": ma2,
            "DMA": dma,
            "Cross": cross,
            "Slope_MA1": slope_ma1,
            "Strong_Slope": strong_slope,
            "Entry": entry,
        }
