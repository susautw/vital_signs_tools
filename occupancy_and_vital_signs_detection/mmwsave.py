import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from tqdm import tqdm

from config_loader import MMWaveConfigLoader
from ovsd import rolling_average_factory, logger as ovsd_logger
from ovsd.dataset import V3Dataset, DatasetDescription
from ovsd.display import AbstractPlotSaver, PlotSaver, PlotCombinedSaver
from ovsd.mmw_info_iter import InvalidFrameHandler
from ovsd.plot import Zone
from ovsd.plot.plots_builder import PlotGroupBuilder, PlotType
from ovsd.plot_configurator.hmap_clim_updater import HMapCLimUpdater
from ovsd.plot_configurator.plot_updater import PlotUpdater
from ovsd.configs import OVSDConfig
from ovsd.structures import init_structures

logger = logging.getLogger("root")
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler(sys.stdout))

MAP_NAME_TO_PLOT_TYPE = {'f': PlotType.FULL_HMAP, 'p': PlotType.POLAR_HMAP}
NANE_STRATEGY_MAP = {
    "ignore": InvalidFrameHandler.Strategy.IGNORE,
    "interpolate": InvalidFrameHandler.Strategy.INTERPOLATION,
    "subtract": InvalidFrameHandler.Strategy.SUBTRACT_FROM_65535,
}


def main(args_=None):
    args = get_arg_parser().parse_args(args_)
    config = OVSDConfig(MMWaveConfigLoader(args.config.read_text().split("\n")))
    init_structures(config)

    ovsd_logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    sources: list[Path]
    source_base_path: Path
    out_base_path: Path = args.out_dir
    if args.source.is_file():
        sources = [args.source]
        source_base_path = args.source.parent
    elif args.source.is_dir():
        sources = sorted(args.source.glob("**/*.bin"))
        source_base_path = args.source
    else:
        raise FileNotFoundError(args.source)

    if len(sources) == 0:
        logger.info("no source processed")
        exit()

    plot_types = [MAP_NAME_TO_PLOT_TYPE[name] for name in args.plot_types] if args.plot_types else []
    zones = set([Zone.get_zone_from_real(*zd) for zd in args.zone] if args.zone else [])

    if len(plot_types) + len(zones) == 0:
        raise RuntimeError("No Plot Types specified")

    plot_builder = PlotGroupBuilder().with_config(config).set_show_rect_in_hmap(False)
    figs = []
    for i, plot_type in enumerate(plot_types):
        fig: plt.Figure = plt.figure(figsize=(4, 4), frameon=False, tight_layout=dict(pad=0))
        plot_builder.add_plot_type(plot_type, fig)
        figs.append(fig)

    for i, zone in enumerate(zones, start=len(plot_types)):
        fig: plt.Figure = plt.figure(figsize=(4, 4), frameon=False, tight_layout=dict(pad=0))
        plot_builder.add_plot_type(PlotType.ZONE_HMAP, fig, zone)
        figs.append(fig)

    ds = V3Dataset(DatasetDescription.get_desc_from_data_path(args.source))

    saver_params = dict(
        base_figs=figs,
        plot=plot_builder.build(),
        frame_configurator=HMapCLimUpdater(rolling_average_factory),
        update_configurator=PlotUpdater(),
        skip=args.skip,
        max_saves=args.max_saves
    )

    saver: AbstractPlotSaver
    if args.combine is None:
        saver = PlotSaver(**saver_params)
    else:
        saver = PlotCombinedSaver(
            **saver_params,
            out_shape=tuple(args.combined_shape),
            search_range=args.search_range,
            aligned=not args.no_align
        )

    invalid_frame_handler_strategy = (
        NANE_STRATEGY_MAP[args.invalid_frame_handler]
        if args.invalid_frame_handler else None
    )

    progress = tqdm(sources)
    try:
        for source in progress:
            progress.set_postfix(file=str(source))
            source_it = ds.get_source_iter(source)
            if invalid_frame_handler_strategy is not None:
                source_it = InvalidFrameHandler(source_it, invalid_frame_handler_strategy)

            out_dir = out_base_path / source.relative_to(source_base_path).with_suffix('')
            out_dir.mkdir(parents=True, exist_ok=True)
            saver.set_context(source_it, out_dir)
            saver.display()
    except KeyboardInterrupt:
        logger.info("interrupted")

    logger.info("done.")


def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="path to source file or directory (should include in a dataset)")
    parser.add_argument("config", type=Path, help="path to config file")
    parser.add_argument("-v", "--verbose", action="store_true")

    parser.add_argument("-o", "--out-dir", type=Path, required=True, help="path to the output directory")
    parser.add_argument("-i", "--invalid-frame-handler", choices=['ignore', 'interpolate', 'subtract'])
    parser.add_argument('-z', '--zone', type=float, nargs=4, action="append", metavar=("r1", "r2", "a1", "a2"))
    parser.add_argument('-f', '--full', dest="plot_types", action="append_const", const="f")
    parser.add_argument('-p', '--polar-full', dest="plot_types", action="append_const", const="p")
    parser.add_argument("--skip", default=50, type=int, help="produces images after the number of frames")
    parser.add_argument("-m", "--max-saves", type=int, help="maximum number of images produced pre file plot")
    sub = parser.add_subparsers(title="combine", dest="combine")
    parser_combine = sub.add_parser("combine")
    parser_combine.add_argument("combined_shape", type=int, action='append', metavar="r", help="row")
    parser_combine.add_argument("combined_shape",  type=int, action='append', metavar="c", help="column")
    parser_combine.add_argument("-sr", "--search-range", type=int, default=10)
    parser_combine.add_argument("--no-align", action="store_true", help="turn off aligning peak")
    return parser


if __name__ == '__main__':
    main()
