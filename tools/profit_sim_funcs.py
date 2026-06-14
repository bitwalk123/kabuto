from datetime import datetime, time

import pandas as pd
from matplotlib import dates as mdates

from pandas import DataFrame, Series


def get_x_range(df: DataFrame) -> tuple[datetime, datetime]:
    dt_date = df.index[0].date()
    dt_open = time(9, 0)
    dt_close = time(15, 30)
    dt_start = datetime.combine(dt_date, dt_open)
    dt_end = datetime.combine(dt_date, dt_close)
    return dt_start, dt_end


def get_y_range(ser: Series):
    y_max = ser.max()
    y_min = ser.min()
    y_pad = (y_max - y_min) * 0.05
    return y_min - y_pad, y_max + y_pad


def to_pd_dt(x):
    return pd.to_datetime(mdates.num2date(x)).tz_localize(None)
