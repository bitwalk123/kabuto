import inspect


class FeatureDefaults:
    """
    特徴量用パラメータ管理クラス
    """
    PERIOD_WARMUP = 300
    PERIOD_MA_1 = 30
    LOSSCUT_1 = -25.0
    N_MINUS_MAX = 90
    DD_PROFIT = 5.0
    DD_RATIO = 0.9

    @classmethod
    def as_dict(cls) -> dict:
        return {
            k: v for k, v in cls.__dict__.items()
            if not k.startswith("_") and not inspect.isroutine(v)
        }
