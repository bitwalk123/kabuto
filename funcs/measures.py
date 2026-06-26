import time
from functools import wraps


def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()

        try:
            return func(*args, **kwargs)

        except Exception as e:
            print(
                f"[ERROR] {func.__name__}: "
                f"{type(e).__name__}: {e}"
            )
            raise

        finally:
            elapsed = time.perf_counter() - start
            print(
                f"[MEASURE] {func.__name__}: "
                f"{elapsed:.6f} sec"
            )

    return wrapper