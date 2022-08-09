from typing import Iterator

import matplotlib.pyplot as plt

from . import IPlotDisplayer
from .. import MMWInfo
from ..plot import IPlot
from ..plot_configurator import PlotConfiguratorPipeline


class PlotShower(IPlotDisplayer):
    _bg_cache = None
    _finalized = False

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
        self._finalized = False
        canvas = self.base_fig.canvas
        canvas.draw()
        self._bg_cache = canvas.copy_from_bbox(self.base_fig.bbox)
        canvas.mpl_connect("close_event", self._finalize)
        try:
            self.base_fig.show()
            for info in self.source_it:
                if self._finalized:
                    break
                self.configurator_pipeline.execute(self.plot, info)
                self.plot.draw()
                canvas.blit(self.base_fig.bbox)
                canvas.flush_events()
        finally:
            if not self._finalized:  # program finished with an exception.
                canvas.close_event()

    def _finalize(self, _event):
        self._finalized = True
