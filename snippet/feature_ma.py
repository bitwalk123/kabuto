from collections import deque

from funcs.technical import MovingAverage


class RealTimeDMA:
    """
    RealTimeDMA:
    リアルタイムで MA1 / MA2 / DMA / クロス / 傾き を逐次更新する軽量クラス。
    FeatureProvider で使用し、ObservationManager に dict を渡す用途を想定。
    """

    def __init__(self, period_1=60, period_2=540, slope_window=5, threshold=0.05):
        self.ma_1 = MovingAverage(window_size=period_1)
        self.ma_2 = MovingAverage(window_size=period_2)
        # ---------------------------------------------------------------------
        self.prev_dma = None
        self.prev_ma_1 = None
        # ---------------------------------------------------------------------
        self.slope_buf = deque(maxlen=slope_window)
        self.threshold = threshold

    def update(self, price):
        ma_1 = self.ma_1.update(price)
        ma_2 = self.ma_2.update(price)
        # ---------------------------------------------------------------------
        # 初期化段階
        # ただし、MovingAverage は None を返さない仕様なので不要かも
        if ma_1 is None or ma_2 is None:
            return None
        # ---------------------------------------------------------------------
        # DMA, delta MA
        dma = ma_1 - ma_2
        # ---------------------------------------------------------------------
        # クロス判定
        if self.prev_dma is None:
            cross = 0
        else:
            if self.prev_dma < 0 and dma > 0:
                cross = +1
            elif self.prev_dma > 0 and dma < 0:
                cross = -1
            else:
                cross = 0
        self.prev_dma = dma
        # ---------------------------------------------------------------------
        # 傾き（diff → rolling mean）
        if self.prev_ma_1 is None:
            diff_ma1 = 0.0
        else:
            diff_ma1 = ma_1 - self.prev_ma_1
        self.prev_ma_1 = ma_1
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
            "MA1": ma_1,
            "MA2": ma_2,
            "DMA": dma,
            "Cross": cross,
            "Slope_MA1": slope_ma1,
            "Strong_Slope": strong_slope,
            "Entry": entry,
        }
