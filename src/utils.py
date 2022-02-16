import contextlib
import inspect
import time


@contextlib.contextmanager
def test_time(name: str = "", line=None):
    if line is None:
        line = inspect.currentframe().f_back.f_back.f_lineno
    start = time.time()
    yield
    print(f"Time {name}:{line=}: {time.time() - start}")
