from beetle.beetle_psar import PSARObject
from structs.posman import PositionType


class PacMan:
    """
    トレンド判定ロジック
    """

    def __init__(self):
        self.counter: int = 0  # トータルカウンター
        self.sar: int = 0  # 反転カウンター
        self.trend: int = 0  # トレンドの向き
        self.epupd_min: int = 1  # トレンド追従を開始するための EP 更新回数の最低回数
        self.has_position: bool = False  # ポジションを持っているか？

    def setTrend(self, ret: PSARObject) -> PositionType:
        # カウンタの更新
        self.counter += 1

        if self.trend != ret.trend:  # トレンド反転時
            if self.trend * ret.trend == -1:  # トレンドが -1 と 1 の時
                self.trend = ret.trend
                self.has_position = False  # ポジションをリセット
                return PositionType.REPAY
            else:  # 最初のトレンド 0 から -1 あるいは 1 へ変更した時
                self.trend = ret.trend
                self.has_position = False  # ポジションをリセット
                return PositionType.NONE
        else:
            if not self.has_position and self.epupd_min < ret.epupd:
                # 建玉無しで EP 更新回数が必要最低回数より大きくなった時
                if 0 < self.trend:
                    self.has_position = True
                    return PositionType.BUY
                elif self.trend < 0:
                    self.has_position = True
                    return PositionType.SELL
                else:
                    self.has_position = False
                    return PositionType.NONE
            else:
                return PositionType.HOLD
