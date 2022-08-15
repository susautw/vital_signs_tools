import argparse
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum, Enum
from pathlib import Path
from typing import Iterator, Callable, Optional

import cv2
import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import QuadMesh

from config_loader import MMWaveConfigLoader
from ovsd import rolling_average_factory
from ovsd.configs import OVSDConfig
from occupancy_and_vital_signs_detection.plots import HeatmapPlotter
from utility import RollingAverage, convert_artist_to_image

args: argparse.Namespace


class MapType(IntEnum):
    Full = 1
    PolarFull = 2
    Zone0 = 4
    Zone1 = 8


class MapSourceType(Enum):
    Full = 'heatmap/full'
    Zone0 = 'heatmap/zone0'
    Zone1 = 'heatmap/zone1'


TYPE_SOURCE_MAP = {
    MapType.Full: MapSourceType.Full,
    MapType.PolarFull: MapSourceType.Full,
    MapType.Zone0: MapSourceType.Zone0,
    MapType.Zone1: MapSourceType.Zone1
}

TYPE_NAME_SUFFIX_MAP = {
    MapType.Full: "full",
    MapType.PolarFull: "polar_full",
    MapType.Zone0: "zone0",
    MapType.Zone1: "zone1"
}

TYPE_MESH_GETTER_MAP = {
    MapType.Full: lambda p: p.full_mesh,
    MapType.PolarFull: lambda p: p.polar_full_mesh,
    MapType.Zone0: lambda p: p.zone_plots[0].mesh,
    MapType.Zone1: lambda p: p.zone_plots[1].mesh
}

ALL_MAP_TYPES = (MapType.Full, MapType.PolarFull, MapType.Zone0, MapType.Zone1)
ALL_SOURCE_TYPES = (MapSourceType.Full, MapSourceType.Zone0, MapSourceType.Zone1)
ZONE_MAP_TYPES = (MapType.Zone0, MapType.Zone1)


def main(args_=None):
    global args
    args = get_arg_parser().parse_args(args_)
    source_path: Path = args.source
    if source_path.is_dir():
        source_paths = sorted(source_path.glob("**/*.h5"))
        source_base_path = source_path
    elif source_path.is_file():
        if source_path.suffix != '.h5':
            raise ValueError(f'{source_path} is not a .h5 file.')
        source_paths = [source_path]
        source_base_path = source_path.parent
    else:
        raise FileNotFoundError(source_path)

    config_path: Path = args.config
    if not config_path.is_file():
        raise FileNotFoundError(f'{config_path}')

    config = OVSDConfig(MMWaveConfigLoader(config_path.read_text().split("\n")))

    if not args.map_types:
        raise ValueError(f'please specify at least one map type')
    map_typ = sum(args.map_types)
    out_base_path: Path = args.output
    if out_base_path.is_file():
        raise FileExistsError(out_base_path)

    for path in source_paths:
        with h5py.File(path) as fp:
            for i, images in enumerate(h5_to_images(fp, config, map_typ, args.show_rect, args.sep_averager)):
                print(f'\r{path}: {i}', end="")
                out_path = out_base_path / path.relative_to(source_base_path).with_suffix(".png")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                for typ, image in images.items():
                    cv2.imwrite(str(out_path.with_stem(f'{out_path.stem}_{TYPE_NAME_SUFFIX_MAP[typ]}_{i}')), image)
            print()
    print("Done!")


InitializeHook = Callable[["HeatmapFigureIterator"], None]
HeatmapSourcesIterator = Iterator[dict[MapSourceType, np.ndarray]]
FigureCollectionFactory = Callable[["HeatmapFigureIterator"], dict[MapType, "FigureCollection"]]
HeatmapSourcesIteratorFactory = Callable[["HeatmapFigureIterator"], HeatmapSourcesIterator]


def h5_to_images(
        fp: h5py.File,
        config: OVSDConfig,
        map_typ: int,
        show_rect: bool,
        sep_averager: bool,
        *,
        initialize_hook: InitializeHook = None,
        skip: int = 0
) -> Iterator[dict[MapType, np.ndarray]]:
    configurator = MeshUpdater()

    if sep_averager:
        configurator = SeparatingAveragingLimitDecorator(configurator, rolling_average_factory)
    else:
        configurator = AveragingLimitDecorator(configurator, rolling_average_factory)

    used_map_types, used_map_source_types = get_map_types_and_sources(map_typ)

    with HeatmapFigureIterator(
            configurator,
            get_figure_collections(used_map_types, config, show_rect),
            get_heatmaps_iter(used_map_source_types, fp, skip)
    ) as heatmap_fig_iter:
        initialize_hook(heatmap_fig_iter)

        for fig_collections in heatmap_fig_iter:
            yield {
                typ: convert_artist_to_image(fig_collection.figure, draw=True)
                for typ, fig_collection in fig_collections.items()
            }


def get_map_types_and_sources(map_typ: int) -> tuple[set[MapType], set[MapSourceType]]:
    used_map_types = {typ for typ in ALL_MAP_TYPES if map_typ & typ}
    used_map_source_types = {TYPE_SOURCE_MAP[typ] for typ in used_map_types}
    used_map_source_types.add(MapSourceType.Full)  # the set must include Full
    return used_map_types, used_map_source_types


@dataclass
class FigureCollection:
    figure: plt.Figure
    ax: plt.Axes
    mesh: QuadMesh = None


class HeatmapFigureIterator(Iterator[dict[MapType, FigureCollection]]):
    _inner_iterator: Iterator[dict[MapType, FigureCollection]] = None

    configurator: "HeatmapConfiguratorBase"
    figure_collections: dict[MapType, FigureCollection]
    config: Optional[OVSDConfig]
    show_rect: bool

    skip: int
    h5_heatmaps: h5py.File
    sources_iter: HeatmapSourcesIterator

    used_map_types: set[MapType]
    used_map_source_types: set[MapSourceType]

    def __init__(
            self,
            configurator: "HeatmapConfiguratorBase",
            figure_collections: dict[MapType, FigureCollection],
            sources_iter: HeatmapSourcesIterator,
    ):
        self.configurator = configurator
        self.figure_collections = figure_collections
        self.sources_iter = sources_iter

    def __next__(self) -> dict[MapType, FigureCollection]:
        if self._inner_iterator is None:
            self._inner_iterator = self._create_inner_iterator()
        return next(self._inner_iterator)

    def _create_inner_iterator(self) -> Iterator[dict[MapType, FigureCollection]]:
        for sources in self.sources_iter:
            self.configurator.configure(sources, self.figure_collections)
            yield self.figure_collections

    def finalize(self) -> None:
        for figure_collection in self.figure_collections.values():
            fig = figure_collection.figure
            fig.clf()
            plt.close(fig)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finalize()


def get_figure_collections(
        used_map_types: set[MapType],
        config: OVSDConfig,
        show_rect: bool
) -> dict[MapType, FigureCollection]:
    figure_collections = {}
    for typ in used_map_types:
        fig = plt.figure()
        ax = fig.add_subplot(111, polar=typ is MapType.PolarFull)
        figure_collections[typ] = FigureCollection(fig, ax)

    # noinspection PyTypeChecker
    plotter = HeatmapPlotter(
        config,
        full=_get_ax_from_figure_collections(figure_collections, MapType.Full),
        polar_full=_get_ax_from_figure_collections(figure_collections, MapType.PolarFull),
        zones={
            i: figure_collections[typ]
            for i, typ in enumerate(ZONE_MAP_TYPES)
            if typ in figure_collections
        }
    )
    plotter.set_show_rect(show_rect)
    plotter.initialize()

    for typ, mesh_getter in TYPE_MESH_GETTER_MAP.items():
        if typ in used_map_types:
            figure_collections[typ].mesh = mesh_getter(plotter)

    return figure_collections


def _get_ax_from_figure_collections(
        figure_collections: dict[MapType, FigureCollection],
        typ: MapType
) -> Optional[plt.Axes]:
    fc = figure_collections.get(typ)
    return fc if fc is None else fc.ax


def get_heatmaps_iter(
        used_map_source_types: set[MapSourceType],
        h5_heatmaps: h5py.File,
        skip: int = 0
) -> HeatmapSourcesIterator:
    sources_typ_mapping = {
        source: np.asarray(h5_heatmaps[source.value])
        for source in used_map_source_types
    }

    for i in range(skip, len(sources_typ_mapping[MapSourceType.Full])):
        yield {typ: sources_typ_mapping[typ][i] for typ in used_map_source_types}


class HeatmapConfiguratorBase(ABC):
    @abstractmethod
    def configure(
            self,
            sources: dict[MapSourceType, np.ndarray],
            figure_collections: dict[MapType, FigureCollection]
    ) -> None: ...


class HeatmapConfiguratorDecorator(HeatmapConfiguratorBase):
    component: HeatmapConfiguratorBase

    def __init__(self, configurator: HeatmapConfiguratorBase):
        self.component = configurator

    def configure(
            self,
            sources: dict[MapSourceType, np.ndarray],
            figure_collections: dict[MapType, FigureCollection]
    ) -> None:
        self.component.configure(sources, figure_collections)


class MeshUpdater(HeatmapConfiguratorBase):

    def configure(
            self,
            sources: dict[MapSourceType, np.ndarray],
            figure_collections: dict[MapType, FigureCollection]
    ) -> None:
        for typ, figure_collection in figure_collections.items():
            figure_collection.mesh.set_array(sources[TYPE_SOURCE_MAP[typ]])


class AveragingLimitDecoratorBase(HeatmapConfiguratorDecorator, ABC):
    def __init__(self, configurator: HeatmapConfiguratorBase, factory: Callable[[], RollingAverage]):
        super().__init__(configurator)
        self._initialize_rolling_averages(factory)

    def configure(
            self,
            sources: dict[MapSourceType, np.ndarray],
            figure_collections: dict[MapType, FigureCollection]
    ) -> None:
        high_bounds = self._get_high_bound(sources)
        for typ, figure_collection in figure_collections.items():
            figure_collection.mesh.set_clim(0, high_bounds[TYPE_SOURCE_MAP[typ]])
        super().configure(sources, figure_collections)

    @abstractmethod
    def _initialize_rolling_averages(self, factory: Callable[[], RollingAverage]) -> None: ...

    @abstractmethod
    def _get_high_bound(self, sources: dict[MapSourceType, np.ndarray]) -> dict[MapSourceType, float]: ...


class AveragingLimitDecorator(AveragingLimitDecoratorBase):
    average: RollingAverage

    def _initialize_rolling_averages(self, factory: Callable[[], RollingAverage]) -> None:
        self.average = factory()

    def _get_high_bound(self, sources: dict[MapSourceType, np.ndarray]) -> dict[MapSourceType, float]:
        bound = np.max(sources[MapSourceType.Full])
        return {typ: bound for typ in sources}


class SeparatingAveragingLimitDecorator(AveragingLimitDecoratorBase):
    averages: dict[MapSourceType, RollingAverage]
    rolling_average_factory: Callable[[], RollingAverage]

    def _initialize_rolling_averages(self, factory: Callable[[], RollingAverage]) -> None:
        self.averages = {}
        self.rolling_average_factory = factory

    def _get_high_bound(self, sources: dict[MapSourceType, np.ndarray]) -> dict[MapSourceType, float]:
        bounds = {}
        for typ, source in sources.items():
            if typ not in self.averages:
                self.averages[typ] = self.rolling_average_factory()
            bounds[typ] = self.averages[typ].next(np.max(source))
        return bounds


def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("-s", "--sep-averager", action="store_true")
    parser.add_argument("-r", "--show-rect", action="store_true")

    parser.add_argument("-f", "--full", action="append_const", const=MapType.Full, dest="map_types")
    parser.add_argument("-p", "--polar-full", action="append_const", const=MapType.PolarFull, dest="map_types")
    parser.add_argument("-z0", "--zone0", action="append_const", const=MapType.Zone0, dest="map_types")
    parser.add_argument("-z1", "--zone1", action="append_const", const=MapType.Zone1, dest="map_types")
    return parser


if __name__ == '__main__':
    main()
