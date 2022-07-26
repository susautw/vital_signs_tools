import contextlib
import ctypes
import inspect
import time
from collections import deque
from math import inf
from typing import Any, Sequence

import cv2
import numpy as np
from matplotlib import pyplot as plt


@contextlib.contextmanager
def test_time(name: str = "", line=None):
    if line is None:
        line = inspect.currentframe().f_back.f_back.f_lineno
    start = time.time()
    yield
    print(f"Time {name}:{line=}: {time.time() - start}")


class RollingAverage:
    values: deque
    window_size: int
    low: float = -inf
    high: float = inf
    init: float = 0
    _rolling_avg: float = 0

    def __init__(self, window_size: int, low: float = None, high: float = None, *, init: float = None):
        if init is not None:
            self.init = init
        self.window_size = window_size
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
        self._rolling_avg = sum(self.values) / self.window_size
        return self._apply_bound(self._rolling_avg)

    def current(self) -> float:
        """
        return rolling average
        """
        return self._apply_bound(self._rolling_avg)

    def _apply_bound(self, value) -> float:
        return min(max(value, self.low), self.high)

    def reset(self) -> None:
        """
        re-init the instance
        """
        self.values = deque([self.init] * self.window_size)
        self._rolling_avg = self.init

    def resize(self, new_window_size: int) -> None:
        """
        resize window of the instance
        """
        num_leak = max(0, new_window_size - self.window_size)
        # slice will ignore ids out of bounds, e.g. (len(([0]*4)[-100:100]) == 4) is True
        self.values = deque([self.init] * num_leak + list(self.values)[0: new_window_size])
        self.window_size = new_window_size

        self._rolling_avg = sum(self.values) / self.window_size


def structure_to_dict(struct: ctypes.Structure) -> dict[str, Any]:
    result = {}
    # noinspection PyProtectedMember
    for field_name, field_typ in struct._fields_:
        result[field_name] = getattr(struct, field_name)
    return result


def convert_fig_to_image(fig: plt.Figure, draw=False) -> np.ndarray:
    if draw:
        fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img = img.reshape((h, w, 3))

    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def combine_images(
        image_size: np.ndarray,
        shape: Sequence[int],
        images: Sequence[np.ndarray],
        n_channels: int
) -> np.ndarray:
    if len(shape) != 2:
        raise ValueError("invalid shape: length should be two.")
    idx_length = np.asarray(shape).prod()
    if len(images) != idx_length:
        raise ValueError("invalid images: length should equal to product of shape")
    w, h = image_size
    wc, hc = shape
    combined_shape = [h * hc, w * wc]
    if n_channels > 1:
        combined_shape.append(n_channels)
    combined_image = np.zeros(combined_shape, dtype=np.uint8)
    for i, img in enumerate(images):
        x, y = image_size * np.unravel_index(i, shape)[::-1]
        combined_image[y: y + h, x: x + w] = img
    return combined_image
