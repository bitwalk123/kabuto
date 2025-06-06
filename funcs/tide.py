import datetime


def conv_datetime_from_timestamp(ts) -> str:
    dt = datetime.datetime.fromtimestamp(ts)
    return f"{dt.year:04}-{dt.month:02}-{dt.day:02} {dt.hour:02}:{dt.minute:02}:{dt.second:02}"
