import datetime


def get_name_15min_chart(code: str, dt: datetime.datetime) -> str:
    year = dt.year
    month = dt.month
    day = dt.day
    return f"{year:4d}/{month:02d}{day:02d}_15min_chart_{code}.png"


def get_name_15min_chart_now(code: str) -> str:
    dt_now = datetime.datetime.now()
    year = dt_now.year
    month = dt_now.month
    day = dt_now.day
    return f"{year:4d}/{month:02d}{day:02d}_15min_chart_{code}.png"
