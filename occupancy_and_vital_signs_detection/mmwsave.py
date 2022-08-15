import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from config_loader import MMWaveConfigLoader
from ovsd.dataset import V3Dataset, DatasetDescription
from ovsd.display import PlotShower, PlotSaver, PlotCombinedSaver
from ovsd.plot import Zone
from ovsd.plot.plots_builder import PlotGroupBuilder, PlotType
from ovsd.plot_configurator import PlotConfiguratorPipeline
from ovsd.plot_configurator.hmap_clim_updater import HMapCLimUpdater
from ovsd.plot_configurator.plot_updater import PlotUpdater
from occupancy_and_vital_signs_detection.h5_to_image import rolling_average_factory
from ovsd.configs import OVSDConfig
from ovsd.structures import init_structures

logger = logging.getLogger("root")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

MAP_NAME_TO_PLOT_TYPE = {'f': PlotType.FULL_HMAP, 'p': PlotType.POLAR_HMAP}


def main(args_=None):
    # args = get_arg_parser().parse_args(args_)
    # p = args.source
    p = Path("ds_lab/data/+15h.packet.20220706T222049.bin")
    config = OVSDConfig(MMWaveConfigLoader(Path('datasets/ds3/vod_vs_68xx_10fps_center.cfg').read_text().split("\n")))
    init_structures(config)

    # plot_types = [MAP_NAME_TO_PLOT_TYPE[name] for name in args.plot_types] if args.plot_types else []
    # zones = set([Zone.get_zone_from_real(*zd) for zd in args.zone] if args.zone else [])

    # len_plots = len(plot_types) + len(zones)
    # if len_plots == 0:
    #     raise RuntimeError("No Plot Types specified")
    #
    # fig: plt.Figure = plt.figure(figsize=(4 * len_plots, 4))
    # gs = fig.add_gridspec(1, len_plots)
    # plot_builder = (PlotGroupBuilder()
    #                 .with_config(config)
    #                 .set_show_rect_in_hmap(args.show_rect)
    #                 )
    # for i, plot_type in enumerate(plot_types):
    #     sub_fig: plt.Figure = fig.add_subfigure(gs[i])
    #     sub_fig.suptitle(plot_type.name)
    #     plot_builder.add_plot_type(plot_type, sub_fig)
    #
    # for i, zone in enumerate(zones, start=len(plot_types)):
    #     sub_fig = fig.add_subfigure(gs[i])
    #     sub_fig.suptitle(
    #         '(r1={:.2f}, r2={:.2f}, a1={:.2f}, a2={:.2f})'.format(*zone.to_real())
    #     )
    #     plot_builder.add_plot_type(PlotType.ZONE_HMAP, sub_fig, zone)

    figs = []
    builder = PlotGroupBuilder().with_config(config).set_show_rect_in_hmap(False)
    fig = plt.figure(figsize=(4, 4), frameon=False, tight_layout=dict(pad=0))
    figs.append(fig)
    builder.add_plot_type(PlotType.FULL_HMAP, fig)
    fig = plt.figure(figsize=(4, 4), frameon=False, tight_layout=dict(pad=0))
    figs.append(fig)
    builder.add_plot_type(PlotType.POLAR_HMAP, fig)

    source_it = V3Dataset(DatasetDescription.get_desc_from_data_path(p)).get_source_iter(p)

    try:
        PlotCombinedSaver(
            figs,
            source_it,
            builder.build(),
            frame_configurator=HMapCLimUpdater(rolling_average_factory),
            update_configurator=PlotUpdater(),
            out_dir=Path("test_out"),
            skip=50,
            search_range=10,
            out_shape=(3, 3)
        ).display()
    except KeyboardInterrupt:
        logger.info("interrupted")

    logger.info("done.")


def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="path to source file (should be placed in a dataset)")
    parser.add_argument("config", type=Path, help="path to config file")
    parser.add_argument("-r", "--show-rect", help="show rectangle around zones", action="store_true")
    parser.add_argument('-z', '--zone', type=float, nargs=4, action="append")
    parser.add_argument('-f', '--full', dest="plot_types", action="append_const", const="f")
    parser.add_argument('-p', '--polar-full', dest="plot_types", action="append_const", const="p")
    return parser


if __name__ == '__main__':
    main()
