import cv2

from . import AbstractPlotConfigurator
from ..plot import plots


class HMapNoiseRemover(AbstractPlotConfigurator):
    def visit_abstract_full_hmap_plot(self, plot: plots.AbstractFullHMapPlot):
        original = self.get_mmw_info().get_full_hmap()
        retain = original[5:]
        _, hmap_mask = cv2.threshold(retain, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        avg_background = retain[hmap_mask == 1].mean()
        print(self.get_mmw_info().get_header().frame_number, avg_background)
        original[:5] = avg_background
