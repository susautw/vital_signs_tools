from typing import Sequence

import numpy as np

from .plot import AbstractZone


class MMWInfo:
    zones: list[AbstractZone]
    # zone_infos: dict[AbstractZone, np.ndarray] = {}  # TODO
    zone_decisions: dict[AbstractZone, bool]
    hmap: np.ndarray

    def __init__(
            self, *,
            zones: Sequence[AbstractZone],
            # zone_infos: dict[AbstractZone, np.ndarray],
            zone_decisions: dict[AbstractZone, bool],
            hmap: np.ndarray
    ):
        self.zones = list(zones)
        # self.zone_infos = zone_infos
        self.zone_decisions = zone_decisions
        self.hmap = hmap

    def get_zone_info(self, zone: AbstractZone) -> np.ndarray:
        raise RuntimeError("Not implemented")

    def get_zone_decision(self, zone: AbstractZone) -> bool:
        return self.zone_decisions[zone]

    def get_zone_hmap(self, zone: AbstractZone) -> np.ndarray:
        return self.hmap[zone.idx_slice]

    def get_full_map(self) -> np.ndarray:
        return self.hmap

    def copy(self) -> "MMWInfo":
        return MMWInfo(zones=self.zones.copy(), zone_decisions=self.zone_decisions.copy(), hmap=self.hmap.copy())