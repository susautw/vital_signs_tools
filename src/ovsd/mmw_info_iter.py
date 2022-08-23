import enum
from typing import Iterator, Callable, Optional

import numpy as np

from . import MMWInfo, logger


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


class InvalidFrameHandler(Iterator[MMWInfo]):
    _inner_iterator = None

    class Strategy(enum.Enum):
        IGNORE = "ignore"
        INTERPOLATION = "interpolation"
        SUBTRACT_FROM_65535 = "subtract"

    def __init__(self, info_iter: Iterator[MMWInfo], strategy: Strategy):
        if strategy is self.Strategy.IGNORE:
            self._inner_iterator = self._ignore_invalid_frames(info_iter)
        elif strategy is self.Strategy.INTERPOLATION:
            self._inner_iterator = self._interpolation_invalid_frames(info_iter)
        elif strategy is self.Strategy.SUBTRACT_FROM_65535:
            self._inner_iterator = self.subtract_from_65535(info_iter)

    def __next__(self) -> MMWInfo:
        return next(self._inner_iterator)

    def _ignore_invalid_frames(self, info_iter: Iterator[MMWInfo]) -> Iterator[MMWInfo]:
        for info in info_iter:
            if self._is_invalid_frame(info):
                logger.debug(f"ignored frame {info.get_header().frame_number}")
            else:
                yield info

    def _interpolation_invalid_frames(self, info_iter: Iterator[MMWInfo]) -> Iterator[MMWInfo]:
        first_valid = None
        invalid_infos = []

        for info in info_iter:
            if self._is_invalid_frame(info):
                invalid_infos.append(info)
            else:
                if invalid_infos:
                    logger.debug(f"interpolate frames from {first_valid.get_header().frame_number} "
                                 f"to {info.get_header().frame_number}")
                    yield from self._interpolation(first_valid, last_valid=info, invalid_infos=invalid_infos)
                    invalid_infos.clear()
                first_valid = info
                yield info

    def _interpolation(
            self,
            first_valid: Optional[MMWInfo],
            last_valid: MMWInfo,
            invalid_infos: list[MMWInfo]
    ) -> list[MMWInfo]:
        if first_valid is None:  # no first valid frame found. ignore those frames.
            return []
        dtype = first_valid.get_full_hmap().dtype
        hmap = np.float32(first_valid.get_full_hmap())
        step = (np.float32(last_valid.get_full_hmap()) - hmap) / (len(invalid_infos) + 1)

        results = []
        for info in invalid_infos:
            hmap += step
            results.append(info.with_hmap(hmap.astype(dtype)))
        return results

    def subtract_from_65535(self, info_iter: Iterator[MMWInfo]) -> Iterator[MMWInfo]:
        for info in info_iter:
            if self._is_invalid_frame(info):
                logger.debug(f"frame {info.get_header().frame_number} is subtract from 65535")
                hmap = info.get_full_hmap().copy()
                mask = hmap > 32000
                hmap[mask] = 65535 - hmap[mask]
                yield info.with_hmap(hmap)
            else:
                yield info

    def _is_invalid_frame(self, info: MMWInfo) -> bool:
        return np.any(info.get_full_hmap() > 32000)
