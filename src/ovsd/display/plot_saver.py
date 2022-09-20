import logging
import sys
from abc import ABC
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np
from matplotlib import pyplot as plt

from iter_utils import pack
from utility import convert_artist_to_image, combine_images
from . import IPlotDisplayer
from .. import MMWInfo
from ..mmw_info_iter import get_peak_aligned_mmw_info_iter
from ..plot import Zone
from ..plot.plot_visitors import IPlotVisitor
from ..plot import plots
from ..plot_configurator import AbstractPlotConfigurator

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
logger.addHandler(logging.StreamHandler(sys.stderr))


class AbstractPlotSaver(IPlotDisplayer, ABC):
    source_it: Iterator[MMWInfo] = None
    out_dir: Path = None  # what dir to save the plots. the dir must exist

    def __init__(
            self,
            base_figs: list[plt.Figure],
            plot: plots.PlotGroup,
            frame_configurator: AbstractPlotConfigurator,
            update_configurator: AbstractPlotConfigurator,
            skip: int = 0,
            max_saves: int = None
    ):
        self.base_figs = base_figs
        self.plot = plot
        self.frame_configurator = frame_configurator
        self.update_configurator = update_configurator
        self.skip = skip
        self.max_saves = max_saves

        self._axes_configurator = _PlotAxesConfigurator()
        self._image_getter = PlotImageGetter()
        self._plot_name_getter = PlotNameGetter()

        self.plot.accept(self._axes_configurator)
        self.plot_name_pairs = []

        for p in self.plot.get_plots():
            p.accept(self._plot_name_getter)
            self.plot_name_pairs.append((self._plot_name_getter.get_result(), p))

        for f in self.base_figs:
            f.canvas.draw()

    def set_context(self, source_it: Iterator[MMWInfo], out_dir: Path):
        self.source_it = source_it
        self.out_dir = out_dir
        if not out_dir.is_dir():
            if out_dir.is_file():
                raise NotADirectoryError(out_dir)
            raise FileNotFoundError(out_dir)
        self.frame_configurator.reset()
        self.update_configurator.reset()

    def display(self) -> None:
        if self.source_it is None or self.out_dir is None:
            raise RuntimeError('set_context before display')


class PlotSaver(AbstractPlotSaver):

    def display(self) -> None:
        super().display()
        saves = 0
        for i, info in enumerate(self.source_it):
            self.frame_configurator.set_mmw_info(info)
            self.plot.accept(self.frame_configurator)
            self.frame_configurator.reset_after_operation()

            if i < self.skip:
                continue

            self.update_configurator.set_mmw_info(info)
            self.plot.accept(self.update_configurator)
            self.update_configurator.reset_after_operation()

            self.plot.draw()
            for f in self.base_figs:
                f.canvas.blit(f.bbox)

            for name, p in self.plot_name_pairs:
                p.accept(self._image_getter)
                img = self._image_getter.get_result()
                self._image_getter.clear()
                fn = info.get_header().frame_number
                if img is None:
                    logger.warning(
                        f"skipped img at frame {fn} with plot {name}"
                    )
                else:
                    out_path = (self.out_dir / self.out_dir.stem)
                    out_path = out_path.with_name(f'{out_path.stem}_{name}_{fn}.png')
                    cv2.imwrite(str(out_path), img)
            saves += 1
            if self.max_saves is not None and saves >= self.max_saves:
                return


class PlotCombinedSaver(AbstractPlotSaver):

    def __init__(
            self,
            base_figs: list[plt.Figure],
            plot: plots.PlotGroup,
            frame_configurator: AbstractPlotConfigurator,
            update_configurator: AbstractPlotConfigurator,
            search_range: int,
            out_shape: tuple[int, int],
            aligned: bool = True,
            skip: int = 0,
            max_saves: int = None
    ):
        self.search_range = search_range
        self.out_shape = out_shape
        self._out_length = int(np.prod(out_shape))
        self.aligned = aligned
        super().__init__(base_figs, plot, frame_configurator, update_configurator, skip, max_saves)

    def display(self) -> None:
        super().display()
        source_it = self.source_it
        if self.aligned:
            source_it = get_peak_aligned_mmw_info_iter(
                self.source_it,
                self.search_range,
                self._out_length,
                self.skip,
                self.update_frame_configurator
            )
        saves = 0
        for i, infos in enumerate(pack(source_it, self._out_length)):
            name_imgs_map = {}
            img_skipped = False
            for info in infos:
                self.update_configurator.set_mmw_info(info)
                self.plot.accept(self.update_configurator)
                self.update_configurator.reset_after_operation()

                self.plot.draw()
                for f in self.base_figs:
                    f.canvas.blit(f.bbox)

                for name, p in self.plot_name_pairs:
                    imgs = name_imgs_map.setdefault(name, [])
                    p.accept(self._image_getter)
                    img = self._image_getter.get_result()
                    self._image_getter.clear()
                    fn = info.get_header().frame_number
                    if img is None:
                        img_skipped = True
                        logger.warning(f"skipped img at frame {fn} with plot {name}")
                        break
                    imgs.append(img)
                if img_skipped:
                    break
            if img_skipped:
                continue
            for name, imgs in name_imgs_map.items():
                out_path = (self.out_dir / self.out_dir.stem)
                out_path = out_path.with_name(f'{out_path.stem}_{name}_combined_{i}.png')
                cv2.imwrite(str(out_path), combine_images(self.out_shape, imgs, 3))
            saves += 1
            if self.max_saves is not None and saves >= self.max_saves:
                return

    def update_frame_configurator(self, info):
        self.frame_configurator.set_mmw_info(info)
        self.plot.accept(self.frame_configurator)
        self.frame_configurator.reset_after_operation()


class PlotImageGetter(IPlotVisitor):
    def __init__(self):
        self.result = None

    def visit_zone_info_plot(self, plot: plots.ZoneInfoPlot):
        raise RuntimeError('ZoneInfoPlot does not implement')

    def visit_plot_group(self, plot: plots.PlotGroup):
        raise RuntimeError('can not get image from a group')

    def visit_abstract_hmap_plot(self, plot: plots.AbstractHMapPlot):
        self.result = convert_artist_to_image(plot.get_axes())

    def visit_polar_hmap_plot(self, plot: plots.PolarHMapPlot):
        assert self.result is not None
        h, _ = self.result.shape[:2]
        h_start = int(h * 0.215)
        h_end = int(h * 0.785)
        self.result = self.result[h_start:h_end]

    def get_result(self) -> Optional[np.ndarray]:
        return self.result

    def clear(self) -> None:
        self.result = None


class PlotNameGetter(IPlotVisitor):
    def __init__(self):
        self.result = None

    def visit_zone_info_plot(self, plot: plots.ZoneInfoPlot):
        raise RuntimeError('ZoneInfoPlot does not implement')

    def visit_plot_group(self, plot: plots.PlotGroup):
        raise RuntimeError('can not get image from a group')

    def visit_zone_hmap_plot(self, plot: plots.ZoneHMapPlot):
        z = Zone(
            plot.zone.range_start,
            plot.zone.range_length,
            plot.zone.azimuth_start,
            plot.zone.azimuth_length,
        ).to_real()
        self.result = f'z_(' + ','.join(f'{x:.2f}' for x in z) + ')'

    def visit_polar_hmap_plot(self, plot: plots.PolarHMapPlot):
        self.result = "polar"

    def visit_full_hmap_plot(self, plot: plots.FullHMapPlot):
        self.result = "full"

    def get_result(self) -> str:
        if self.result is None:
            raise RuntimeError("No result")
        return self.result


class _PlotAxesConfigurator(AbstractPlotConfigurator):
    def visit_abstract_hmap_plot(self, plot: plots.AbstractHMapPlot):
        plot.get_axes().set_axis_off()

    def reset(self) -> None: ...
