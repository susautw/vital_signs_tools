from typing import Iterator, Optional

import numpy as np
from matplotlib import pyplot as plt

from utility import convert_artist_to_image
from . import IPlotDisplayer
from .. import MMWInfo
from ..plot import plots
from ..plot.plot_visitors import IPlotVisitor
from ..plot.plots import PlotGroup
from ..plot_configurator import PlotConfiguratorPipeline, AbstractPlotConfigurator


class PlotSaver(IPlotDisplayer):
    _bg_caches: list

    def __init__(
            self,
            base_figs: list[plt.Figure],
            source_it: Iterator[MMWInfo],
            plot: PlotGroup,
            configurator_pipeline: PlotConfiguratorPipeline
    ):
        self.base_figs = base_figs
        self.source_it = source_it
        self.plot = plot
        self.configurator_pipeline = configurator_pipeline

        self._bg_caches = []
        for f in self.base_figs:
            f.canvas.draw()
            self._bg_caches.append(f.canvas.copy_from_bbox(f.bbox))

    def display(self) -> None:
        for info in self.source_it:
            self.configurator_pipeline.execute(self.plot, info)
            self.plot.draw()
            for f in self.base_figs:
                f.canvas.blit(f.bbox)


class PlotImageGetter(IPlotVisitor):
    def __init__(self):
        self.result = None

    def visit_zone_info_plot(self, plot: plots.ZoneInfoPlot):
        raise RuntimeError('ZoneInfoPlot does not implement')

    def visit_plot_group(self, plot: plots.PlotGroup):
        raise RuntimeError('can not get image from a group')

    def visit_abstract_hmap_plot(self, plot: plots.AbstractHMapPlot):
        self.result = convert_artist_to_image(plot.get_axes())

    def get_result(self) -> Optional[np.ndarray]:
        return self.result


class _PlotConfigurator(AbstractPlotConfigurator):
    def visit_abstract_hmap_plot(self, plot: plots.AbstractHMapPlot):
        plot.get_axes().set_axis_off()
