from . import AbstractPlotConfigurator
from ..plot import plots


class PlotUpdater(AbstractPlotConfigurator):
    def visit_zone_hmap_plot(self, plot: plots.ZoneHMapPlot):
        plot.set_data(self.get_mmw_info().get_zone_hmap(plot.zone))

    def visit_abstract_full_hmap_plot(self, plot: plots.AbstractFullHMapPlot):
        info = self.get_mmw_info()
        plot.set_data((info.get_full_map(), {z: info.get_zone_decision(z) for z in plot.rect_zones}))
