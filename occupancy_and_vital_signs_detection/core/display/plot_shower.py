from typing import Iterator

import matplotlib.pyplot as plt

from . import IPlotDisplayer
from .. import MMWInfo
from ..plot import IPlot
from ..plot_configurator import PlotConfiguratorPipeline


class PlotShower(IPlotDisplayer):
    _bg_cache = None

    def __init__(
            self,
            base_fig: plt.Figure,
            source_it: Iterator[MMWInfo],
            plot: IPlot,
            configurator_pipeline: PlotConfiguratorPipeline
    ):
        self.base_fig = base_fig
        self.source_it = source_it
        self.plot = plot
        self.configurator_pipeline = configurator_pipeline

    def display(self) -> None:
        canvas = self.base_fig.canvas
        canvas.draw()
        self._bg_cache = canvas.copy_from_bbox(self.base_fig.bbox)
        for info in self.source_it:
            self.configurator_pipeline.execute(self.plot, info)
            self.plot.draw()
            canvas.blit(self.base_fig.bbox)
            canvas.flush_events()
