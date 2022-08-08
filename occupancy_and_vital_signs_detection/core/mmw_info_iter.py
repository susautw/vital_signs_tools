from typing import Iterator

import numpy as np

from . import MMWInfo


class HMapOnlyMMWInfoIterator(Iterator[MMWInfo]):
    hmaps: np.ndarray
    i: int

    def __init__(self, hmaps: np.ndarray):
        self.hmaps = hmaps
        self.i = 0

    def __next__(self) -> MMWInfo:
        if self.i >= len(self.hmaps):
            raise StopIteration()
        info = MMWInfo(
            zones=[],
            zone_decisions={},
            hmap=self.hmaps[self.i]
        )
        self.i += 1
        return info
