from typing import Callable

import numpy as np

from utility import RollingAverage
from . import AbstractPlotConfigurator
from .. import MMWInfo
from ..plot import plots


class HMapCLimUpdater(AbstractPlotConfigurator):
    rolling_average: RollingAverage

    def __init__(self, rolling_average_factory: Callable[[], RollingAverage]):
        self.rolling_average = rolling_average_factory()

    def set_mmw_info(self, info: MMWInfo):
        super().set_mmw_info(info)
        self.rolling_average.next(np.max(info.get_full_map()))

    def visit_abstract_hmap_plot(self, plot: plots.AbstractHMapPlot):
        plot.set_clim(0, self.rolling_average.current())
