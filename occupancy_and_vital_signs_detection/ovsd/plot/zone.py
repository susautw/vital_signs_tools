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


@dataclass(frozen=True, unsafe_hash=False)
class Zone(AbstractZone):
    range_start: int
    range_length: int
    azimuth_start: int
    azimuth_length: int
