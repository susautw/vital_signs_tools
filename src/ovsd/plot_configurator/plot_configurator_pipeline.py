from . import AbstractPlotConfigurator
from .. import MMWInfo
from ..plot import plots


class PlotConfiguratorPipeline:
    _configurators: list[AbstractPlotConfigurator]

    def __init__(self, *configurators: AbstractPlotConfigurator):
        self._configurators = [*configurators]

    def insert_configurator(self, i: int, configurator: AbstractPlotConfigurator):
        self._configurators.insert(i, configurator)

    def add_configurator(self, configurator: AbstractPlotConfigurator):
        self._configurators.append(configurator)

    def execute(self, plot: plots.IPlot, info: MMWInfo):
        for c in self._configurators:
            c.set_mmw_info(info)
            plot.accept(c)
