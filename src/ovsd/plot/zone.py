from abc import ABC
from dataclasses import dataclass
from functools import cache, cached_property


class AbstractZone(ABC):
    range_start: int
    range_length: int
    azimuth_start: int
    azimuth_length: int

    @cache
    def order(self) -> tuple[float, float]:
        return self.azimuth_start + self.azimuth_length // 2, self.range_start + self.range_length // 2

    @cached_property
    def idx_slice(self) -> tuple[slice, slice]:
        return (
            slice(self.range_start, self.range_start + self.range_length),
            slice(self.azimuth_start, self.azimuth_start + self.azimuth_length)
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, AbstractZone):
            return False
        return all(
            getattr(self, var_name) == getattr(other, var_name)
            for var_name in AbstractZone.__annotations__
        )

    def __hash__(self):
        result = 3
        for var_name in AbstractZone.__annotations__:
            result = 37 * result + hash(getattr(self, var_name))
        return result


@dataclass(frozen=True, unsafe_hash=False, )
class Zone(AbstractZone):
    range_start: int
    range_length: int
    azimuth_start: int
    azimuth_length: int

    def to_real(self) -> tuple[float, float, float, float]:
        from ovsd import structures
        angle_unit = 120 / structures.config.num_angle_bins
        range_unit = 3 / structures.config.num_range_bins
        return (
            self.range_start * range_unit,
            (self.range_start + self.range_length) * range_unit,
            self.azimuth_start * angle_unit - 60,
            (self.azimuth_start + self.azimuth_length) * angle_unit - 60,
        )

    # DO NOT Remove this statement
    #  dataclass does not check the super class's __hash__
    __hash__ = AbstractZone.__hash__

    @classmethod
    def get_zone_from_real(cls, r1: float, r2: float, a1: float, a2: float) -> "Zone":
        from ovsd import structures

        if not (r2 > r1 and a2 > a1):
            raise ValueError(f"invalid zone value {r1=} {r2=}, {a1=}, {a2=}")
        for v, lim_low, lim_high in [
            (r1, 0, 3),
            (r2, 0, 3),
            (a1, -60, 60),
            (a2, -60, 60),
        ]:
            if not lim_low <= v <= lim_high:
                raise ValueError(f"invalid zone value {r1=} {r2=}, {a1=}, {a2=}")
        angle_unit = 120 / structures.config.num_angle_bins
        range_unit = 3 / structures.config.num_range_bins
        a1 += 60
        a2 += 60

        return Zone(
            int(r1 // range_unit),
            int((r2 - r1) // range_unit),
            int(a1 // angle_unit),
            int((a2 - a1) // angle_unit)
        )
