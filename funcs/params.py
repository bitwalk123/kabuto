def get_default_psar_params() -> dict:
    """
    デフォルトの Parabolic SAR 関連のパラメータを返す関数
    :return:
    """
    dict_psar = dict()

    # for Parabolic SAR
    dict_psar["af_init"]: float = 0.00001
    dict_psar["af_step"]: float = 0.00001
    dict_psar["af_max"]: float = 0.01
    # for Trend Chaser
    dict_psar["factor_d"]: float = 25  # 許容される ys と PSAR の最大差異
    dict_psar["factor_c"]: float = 0.95  # ys と psar の間を縮める係数

    # for Smoothing Spline
    dict_psar["power_lam"]: int = 6  # Lambda for smoothing spline
    dict_psar["n_smooth_min"]: int = 60  # dead time (min) at start up
    dict_psar["n_smooth_max"]: int = 600  # maximum data for smoothing

    return dict_psar
