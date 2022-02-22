import contextlib
import inspect
import time
from collections import deque
from math import inf


@contextlib.contextmanager
def test_time(name: str = "", line=None):
    if line is None:
        line = inspect.currentframe().f_back.f_back.f_lineno
    start = time.time()
    yield
    print(f"Time {name}:{line=}: {time.time() - start}")


class RollingAverage:
    values: deque
    length: int
    low: float = -inf
    high: float = inf
    init: float = 0
    _rolling_avg: float = 0

    def __init__(self, length: int, low: float = None, high: float = None, *, init: float = None):
        if init is not None:
            self.init = init
        self.length = length
        if low is not None:
            self.low = low
        if high is not None:
            self.high = high
        self.reset()

    def next(self, new_value: float) -> float:
        """
        set new value and return new rolling average
        """
        self.values.popleft()
        self.values.append(new_value)
        self._rolling_avg = sum(self.values) / self.length
        return self._apply_bound(self._rolling_avg)

    def current(self) -> float:
        """
        return rolling average
        """
        return self._apply_bound(self._rolling_avg)

    def _apply_bound(self, value) -> float:
        return min(max(value, self.low), self.high)

    def reset(self) -> None:
        self.values = deque([self.init] * self.length)
        self._rolling_avg = self.init

    def resize(self, new_length: int) -> None:
        num_leak = max(0, new_length - self.length)
        # slice will ignore ids out of bounds, e.g. (len(([0]*4)[-100:100]) == 4) is True
        self.values = deque([self.init] * num_leak + list(self.values)[0: new_length])
        self.length = new_length

        self._rolling_avg = sum(self.values) / self.length
