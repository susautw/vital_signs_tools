from . import AbstractPlotConfigurator
from .. import MMWInfo
from ..plot import plots, IPlot


class PlotConfiguratorPipeline(AbstractPlotConfigurator):
    _configurators: list[AbstractPlotConfigurator]
    _executed: bool

    def __init__(self, *configurators: AbstractPlotConfigurator):
        self._configurators = [*configurators]
        self._executed = False

    def insert_configurator(self, i: int, configurator: AbstractPlotConfigurator):
        self._configurators.insert(i, configurator)

    def add_configurator(self, configurator: AbstractPlotConfigurator):
        self._configurators.append(configurator)

    def _execute(self, plot: IPlot):
        if self._executed:
            return
        self.execute(plot, self.get_mmw_info())

    def execute(self, plot: plots.IPlot, info: MMWInfo):
        for c in self._configurators:
            c.set_mmw_info(info)
            plot.accept(c)
        self._executed = True

    def reset(self) -> None:
        self._executed = False
        for c in self._configurators:
            c.reset()

    def visit_zone_info_plot(self, plot: plots.ZoneInfoPlot):
        self._execute(plot)

    def visit_plot_group(self, plot: plots.PlotGroup):
        self._execute(plot)

    def visit_abstract_hmap_plot(self, plot: plots.AbstractHMapPlot):
        self._execute(plot)

    def visit_zone_hmap_plot(self, plot: plots.ZoneHMapPlot):
        self._execute(plot)

    def visit_abstract_full_hmap_plot(self, plot: plots.AbstractFullHMapPlot):
        self._execute(plot)

    def visit_polar_hmap_plot(self, plot: plots.PolarHMapPlot):
        self._execute(plot)

    def visit_full_hmap_plot(self, plot: plots.FullHMapPlot):
        self._execute(plot)
