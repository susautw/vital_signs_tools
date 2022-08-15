from abc import ABC
from typing import Optional

from .. import MMWInfo
from ..plot import plots
from ..plot.plot_visitors import IPlotVisitor


class AbstractPlotConfigurator(IPlotVisitor, ABC):
    _mmw_info: Optional[MMWInfo] = None

    def visit_zone_info_plot(self, plot: plots.ZoneInfoPlot):
        pass

    def visit_plot_group(self, plot: plots.PlotGroup):
        for p in plot.get_plots():
            p.accept(self)

    def visit_abstract_hmap_plot(self, plot: plots.AbstractHMapPlot):
        pass

    def visit_zone_hmap_plot(self, plot: plots.ZoneHMapPlot):
        pass

    def visit_abstract_full_hmap_plot(self, plot: plots.AbstractFullHMapPlot):
        pass

    def visit_polar_hmap_plot(self, plot: plots.PolarHMapPlot):
        pass

    def visit_full_hmap_plot(self, plot: plots.FullHMapPlot):
        pass

    def get_mmw_info(self) -> MMWInfo:
        return self._mmw_info

    def set_mmw_info(self, info: MMWInfo):
        self._mmw_info = info

    def reset_after_operation(self) -> None:
        self._mmw_info = None

    def reset(self) -> None:
        pass
