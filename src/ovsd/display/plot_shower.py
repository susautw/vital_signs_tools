import time
from typing import Iterator

import matplotlib.pyplot as plt

from . import IPlotDisplayer
from .. import MMWInfo
from ..plot import IPlot
from ..plot_configurator import AbstractPlotConfigurator


class PlotShower(IPlotDisplayer):
    _bg_cache = None
    _finalized = False

    def __init__(
            self,
            base_fig: plt.Figure,
            source_it: Iterator[MMWInfo],
            plot: IPlot,
            configurator: AbstractPlotConfigurator,
            delay: float,
            mute: bool = False
    ):
        self.base_fig = base_fig
        self.source_it = source_it
        self.plot = plot
        self.configurator = configurator
        self.delay = delay
        self.mute = mute

    def display(self) -> None:
        self._finalized = False
        canvas = self.base_fig.canvas
        canvas.draw()
        original_handler = None
        if self.mute:
            original_handler = canvas.callbacks.exception_handler
            canvas.callbacks.exception_handler = self.muted_exc_handler
        canvas.mpl_connect("close_event", self._finalize)
        try:
            self.base_fig.show()
            for info in self.source_it:
                if self._finalized:
                    break

                self.configurator.set_mmw_info(info)
                self.plot.accept(self.configurator)
                self.configurator.reset()

                self.plot.draw()
                canvas.blit(self.base_fig.bbox)
                canvas.flush_events()
                time.sleep(self.delay)
        finally:
            if not self._finalized:  # program finished with an exception.
                canvas.close_event()
            if self.mute:
                canvas.callbacks.exception_handler = original_handler

    def _finalize(self, _event):
        self._finalized = True

    def muted_exc_handler(self, _exc):
        pass
