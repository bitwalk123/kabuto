import inspect


class FeatureDefaults:
    """
    特徴量用パラメータ管理クラス
    """
    PERIOD_WARMUP = 300
    PERIOD_MA_1 = 30
    PERIOD_RSI = 150
    PERIOD_MOM = 300
    N_MINUS_MAX = 300
    LOSSCUT_1 = -25.0
    DD_RATIO_MAX = 0.75
    DD_THRESHOLD = 10.0

    @classmethod
    def as_dict(cls) -> dict:
        return {
            k: v for k, v in cls.__dict__.items()
            if not k.startswith("_") and not inspect.isroutine(v)
        }
