import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from config_loader import MMWaveConfigLoader
from ovsd import rolling_average_factory, logger as ovsd_logger
from ovsd.dataset import V3Dataset, DatasetDescription
from ovsd.display import PlotShower
from ovsd.mmw_info_iter import InvalidFrameHandler
from ovsd.plot import Zone
from ovsd.plot.plots_builder import PlotGroupBuilder, PlotType
from ovsd.plot_configurator import PlotConfiguratorPipeline
from ovsd.plot_configurator.hmap_clim_updater import HMapCLimUpdater
from ovsd.plot_configurator.plot_updater import PlotUpdater
from ovsd.configs import OVSDConfig
from ovsd.structures import init_structures

logger = logging.getLogger("root")
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler(sys.stdout))

MAP_NAME_TO_PLOT_TYPE = {'f': PlotType.FULL_HMAP, 'p': PlotType.POLAR_HMAP}
MAP_PLOT_TYPE_TO_FIG_TITLE = {
    PlotType.FULL_HMAP: "Heart Tracking",
    PlotType.POLAR_HMAP: "Radar Return"
}

NANE_STRATEGY_MAP = {
    "ignore": InvalidFrameHandler.Strategy.IGNORE,
    "interpolate": InvalidFrameHandler.Strategy.INTERPOLATION,
    "subtract": InvalidFrameHandler.Strategy.SUBTRACT_FROM_65535,
}


def main(args_=None):
    args = get_arg_parser().parse_args(args_)

    ovsd_logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    p = args.source
    config = OVSDConfig(MMWaveConfigLoader(args.config.read_text().split("\n")))
    init_structures(config)

    plot_types = [MAP_NAME_TO_PLOT_TYPE[name] for name in args.plot_types] if args.plot_types else []
    zds = args.zone if isinstance(args.zone[0], list) else [args.zone] if args.zone is not None else []
    zones = set([Zone.get_zone_from_real(*zd) for zd in zds])

    len_plots = len(plot_types) + len(zones)
    if len_plots == 0:
        raise RuntimeError("No Plot Types specified")

    fig: plt.Figure = plt.figure(figsize=(4 * len_plots, 4))
    fig.canvas.manager.set_window_title("Heart Radar")
    gs = fig.add_gridspec(1, len_plots)
    plot_builder = (PlotGroupBuilder()
                    .with_config(config)
                    .set_show_rect_in_hmap(args.show_rect)
                    )
    for i, plot_type in enumerate(plot_types):
        sub_fig: plt.Figure = fig.add_subfigure(gs[i])
        sub_fig.suptitle(MAP_PLOT_TYPE_TO_FIG_TITLE[plot_type])
        plot_builder.add_plot_type(plot_type, sub_fig)

    for i, zone in enumerate(zones, start=len(plot_types)):
        sub_fig = fig.add_subfigure(gs[i])
        sub_fig.suptitle('Cardiac Cycle Image')
        plot_builder.add_plot_type(PlotType.ZONE_HMAP, sub_fig, zone)

    pipeline = PlotConfiguratorPipeline(
        HMapCLimUpdater(rolling_average_factory),
        PlotUpdater()
    )

    source_it = V3Dataset(DatasetDescription.get_desc_from_data_path(p)).get_source_iter(p)

    if args.invalid_frame_handler:
        logger.info(f"Invalid frame handler enabled: {args.invalid_frame_handler}")
        source_it = InvalidFrameHandler(source_it, NANE_STRATEGY_MAP[args.invalid_frame_handler])

    try:
        PlotShower(fig, source_it, plot_builder.build(), pipeline, delay=0.107, mute=True).display()
    except KeyboardInterrupt:
        logger.info("interrupted")

    logger.info("done.")


def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="path to source file (should include in a dataset)")
    parser.add_argument("config", type=Path, help="path to config file")
    parser.add_argument("-v", "--verbose", action="store_true")

    parser.add_argument("-r", "--show-rect", help="show rectangle around zones", action="store_true")
    parser.add_argument("-i", "--invalid-frame-handler", choices=['ignore', 'interpolate', 'subtract'])

    # NOTE: arg "zone" can accept multiple zones by set action to append
    parser.add_argument('-z', '--zone', type=float, nargs=4, metavar=("r1", "r2", "a1", "a2"))
    parser.add_argument('-f', '--full', dest="plot_types", action="append_const", const="f")
    parser.add_argument('-p', '--polar-full', dest="plot_types", action="append_const", const="p")
    return parser


if __name__ == '__main__':
    main()
