from typing import Callable, Optional

import numpy as np

from utility import RollingAverage
from . import AbstractPlotConfigurator
from .. import MMWInfo
from ..plot import plots, AbstractZone


class HMapCLimRAUpdater(AbstractPlotConfigurator):
    """
    Update All HMap plot's clim using a RollingAverage.
    """
    rolling_average: RollingAverage

    def __init__(self, rolling_average_factory: Callable[[], RollingAverage]):
        self.rolling_average = rolling_average_factory()

    def set_mmw_info(self, info: MMWInfo):
        super().set_mmw_info(info)
        self.rolling_average.next(np.max(info.get_full_hmap()))

    def visit_abstract_hmap_plot(self, plot: plots.AbstractHMapPlot):
        plot.set_clim(0, self.rolling_average.current())

    def reset(self) -> None:
        self.rolling_average.reset()


class HMapCLimSepRAUpdater(AbstractPlotConfigurator):
    """
    Update All HMap plot's clim using separate RollingAverages.
    """
    rolling_average_map: dict[Optional[AbstractZone], RollingAverage]  #: None representing FullMap
    factory: Callable[[], RollingAverage]

    def __init__(self, rolling_average_factory: Callable[[], RollingAverage]):
        self.rolling_average_map = {}
        self.factory = rolling_average_factory

    def set_mmw_info(self, info: MMWInfo):
        super().set_mmw_info(info)
        for zone, ra in self.rolling_average_map.items():
            self._update_ra(ra, info, zone)

    def visit_zone_hmap_plot(self, plot: plots.ZoneHMapPlot):
        plot.set_clim(0, self._get_initialized_ra(plot.zone).current())

    def visit_abstract_full_hmap_plot(self, plot: plots.AbstractFullHMapPlot):
        plot.set_clim(0, self._get_initialized_ra(None).current())

    def _update_ra(self, ra: RollingAverage, info: MMWInfo, zone: Optional[AbstractZone]):
        if zone is None:
            val_source = info.get_full_hmap()
        else:
            val_source = info.get_zone_hmap(zone)

        ra.next(np.max(val_source))

    def _get_initialized_ra(self, zone: Optional[AbstractZone]) -> RollingAverage:
        if zone in self.rolling_average_map:
            return self.rolling_average_map[zone]
        else:
            ra = self.rolling_average_map[zone] = self.factory()
            self._update_ra(ra, self.get_mmw_info(), zone)
            return ra

    def reset(self) -> None:
        self.rolling_average_map.clear()


class HMapCLimFixedUpdater(AbstractPlotConfigurator):
    def __init__(self, vmin: float, vmax: float):
        self.vmin = vmin
        self.vmax = vmax

    def visit_abstract_hmap_plot(self, plot: plots.AbstractHMapPlot):
        plot.set_clim(self.vmin, self.vmax)
