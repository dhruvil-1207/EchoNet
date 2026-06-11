import time
from functools import wraps
from .logger import log_performance


def profile(func):

    @wraps(func)
    def wrapper(*args, **kwargs):

        start = time.perf_counter()

        result = func(*args, **kwargs)

        duration = time.perf_counter() - start

        log_performance(
            func.__name__,
            duration,
            input_samples=len(args[0])
            if len(args) > 0 and hasattr(args[0], "__len__")
            else None
        )

        return result

    return wrapper