from typing import Iterator, Callable

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
            header=None,
            zone_decisions={},
            zone_infos={},
            hmap=self.hmaps[self.i]
        )
        self.i += 1
        return info


def get_peak_aligned_mmw_info_iter(
        info_iter: Iterator[MMWInfo],
        search_range: int,
        out_length: int,
        skip: int = 0,
        hook: Callable[[MMWInfo], None] = None,
):
    infos = []
    search_len = 0
    remaining_length = 0
    for i, info in enumerate(info_iter):
        if hook is not None:
            hook(info)
        if i < skip:
            continue
        if remaining_length > 0:
            yield info
            remaining_length -= 1
            continue
        if search_len < search_range:
            infos.append(info)
            search_len += 1
        else:
            max_idx = np.asarray(
                [info.get_full_hmap() for info in infos]
            ).sum(axis=(1, 2)).argmax()
            if search_len - max_idx >= out_length:
                yield from infos[max_idx: max_idx + out_length]
                infos = infos[max_idx + out_length:]
                search_len = len(infos)
            else:
                yield from infos[max_idx:]
                remaining_length = out_length - (search_len - max_idx)
                search_len = 0
                infos = []
