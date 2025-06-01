import datetime


def get_time_range_today():
    dt = datetime.datetime.now()
    dt_start = datetime.datetime(dt.year, dt.month, dt.day, hour=9, minute=0)
    dt_end = datetime.datetime(dt.year, dt.month, dt.day, hour=15, minute=25)
    ts_start = dt_start.timestamp()
    ts_end = dt_end.timestamp()
    return ts_start, ts_end
