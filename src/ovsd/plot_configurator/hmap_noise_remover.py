from . import AbstractPlotConfigurator
from ..plot import plots


# TODO
class HMapNoiseRemover(AbstractPlotConfigurator):
    def visit_abstract_full_hmap_plot(self, plot: plots.AbstractFullHMapPlot):
        super().visit_abstract_full_hmap_plot(plot)