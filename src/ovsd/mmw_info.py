from typing import Optional

import numpy as np

from tlv import FrameHeader
from .plot import AbstractZone
from .structures import VitalSignsVectorTLV


class MMWInfo:
    header: FrameHeader
    zone_infos: dict[AbstractZone, VitalSignsVectorTLV]
    zone_decisions: dict[AbstractZone, bool]
    hmap: np.ndarray

    def __init__(
            self, *,
            header: Optional[FrameHeader],
            zone_infos: dict[AbstractZone, VitalSignsVectorTLV],
            zone_decisions: dict[AbstractZone, bool],
            hmap: np.ndarray
    ):
        self.header = header
        self.zone_infos = zone_infos
        self.zone_decisions = zone_decisions
        self.hmap = hmap

    def get_header(self) -> FrameHeader:
        if self.header is None:
            raise RuntimeError("No header provided")
        return self.header

    def get_zone_info(self, zone: AbstractZone) -> Optional[VitalSignsVectorTLV]:
        return self.zone_infos.get(zone)

    def get_zone_decision(self, zone: AbstractZone) -> bool:
        return self.zone_decisions.get(zone, False)

    def get_zone_hmap(self, zone: AbstractZone) -> np.ndarray:
        return self.hmap[zone.idx_slice]

    def get_full_hmap(self) -> np.ndarray:
        return self.hmap

    def copy(self) -> "MMWInfo":
        return MMWInfo(
            header=self.header,
            zone_infos=self.zone_infos.copy(),
            zone_decisions=self.zone_decisions.copy(),
            hmap=self.hmap.copy()
        )

    def with_hmap(self, hmap: np.ndarray) -> "MMWInfo":
        return MMWInfo(
            header=self.header,
            zone_infos=self.zone_infos,
            zone_decisions=self.zone_decisions,
            hmap=hmap
        )
