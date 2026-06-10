import inspect


class FeatureDefaults:
    """
    特徴量用パラメータ管理クラス
    """
    PERIOD_WARMUP = 300
    PERIOD_MA_1 = 30
    PERIOD_MA_2 = 900
    PERIOD_MOM = 300
    N_MINUS_MAX = 900
    LOSSCUT_1 = -50.0
    TRAILING_THRESHOLD = 40

    @classmethod
    def as_dict(cls) -> dict:
        return {
            k: v for k, v in cls.__dict__.items()
            if not k.startswith("_") and not inspect.isroutine(v)
        }
