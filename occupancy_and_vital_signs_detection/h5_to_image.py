import argparse
from enum import IntEnum, Enum
from pathlib import Path
from typing import Union, Iterator, Callable

import cv2
import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import QuadMesh

from config_loader import MMWaveConfigLoader
from occupancy_and_vital_signs_detection.main import Config
from occupancy_and_vital_signs_detection.plots import HeatmapPlotter
from utility import RollingAverage, convert_fig_to_image

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

Averager = Union[RollingAverage, dict[MapSourceType, RollingAverage]]


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

    config = Config(MMWaveConfigLoader(config_path.read_text().split("\n")))

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


InitializeHook = Callable[[dict[MapType, plt.Figure], dict[MapType, plt.Axes]], None]


def h5_to_images(
        fp: h5py.File,
        config: Config,
        map_typ: int,
        show_rect: bool,
        sep_averager: bool,
        *,
        initialize_hook: InitializeHook = None,
        skip: int = 0
) -> Iterator[dict[MapType, np.ndarray]]:
    figs: dict[MapType, plt.Figure] = {}
    axs: dict[MapType, plt.Axes] = {}
    meshes: dict[MapType, QuadMesh] = {}
    used_map_types: set = set()

    try:
        for typ in ALL_MAP_TYPES:
            if map_typ & typ:
                figs[typ] = plt.figure()
                axs[typ] = figs[typ].add_subplot(111, polar=typ is MapType.PolarFull)
                used_map_types.add(typ)

        if initialize_hook is not None:
            initialize_hook(figs, axs)

        plotter = HeatmapPlotter(
            config,
            full=axs.get(MapType.Full),
            polar_full=axs.get(MapType.PolarFull),
            zones={i: axs[typ] for i, typ in enumerate(ZONE_MAP_TYPES) if typ in axs}
        )
        plotter.set_show_rect(show_rect)
        plotter.initialize()

        for typ, mesh_getter in TYPE_MESH_GETTER_MAP.items():
            if typ in used_map_types:
                meshes[typ] = mesh_getter(plotter)

        used_map_source_types: set[MapSourceType] = {TYPE_SOURCE_MAP[typ] for typ in used_map_types}

        averager_params = dict(window_size=4, low=1000, init=0)
        averager: Averager
        if sep_averager:
            averager = {map_source: RollingAverage(**averager_params) for map_source in used_map_source_types}
        else:
            averager = RollingAverage(**averager_params)

        sources: dict[MapSourceType, np.ndarray] = {source: np.asarray(fp[source.value]) for source in ALL_SOURCE_TYPES}

        for i in range(skip, len(sources[MapSourceType.Full])):
            high_bounds = _update_averager(averager, sources, i, sep_averager)
            result_images = {}
            for typ, mesh in meshes.items():
                source_typ = TYPE_SOURCE_MAP[typ]
                mesh.set_clim(0, high_bounds[source_typ])
                mesh.set_array(sources[source_typ][i])
                result_images[typ] = convert_fig_to_image(figs[typ], draw=True)
            yield result_images
    finally:
        for fig in figs.values():
            fig.clf()
            plt.close(fig)


def _update_averager(
        averager: Averager,
        sources: dict[MapSourceType, np.ndarray],
        source_idx: int,
        sep_averager: bool
) -> dict[MapSourceType, float]:
    result = {}
    if sep_averager:
        for source_typ, source in sources.items():
            result[source_typ] = averager[source_typ].next(np.max(source[source_idx]))
    else:
        value = averager.next(np.max(sources[MapSourceType.Full][source_idx]))
        for source_typ in sources:
            result[source_typ] = value
    return result


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
