import logging
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from config_loader import MMWaveConfigLoader
from occupancy_and_vital_signs_detection.core.display.plot_shower import PlotShower
from occupancy_and_vital_signs_detection.core.mmw_info_iter import HMapOnlyMMWInfoIterator
from occupancy_and_vital_signs_detection.core.plot.plots_builder import PlotGroupBuilder, PlotType
from occupancy_and_vital_signs_detection.core.plot_configurator import PlotConfiguratorPipeline
from occupancy_and_vital_signs_detection.core.plot_configurator.hmap_clim_updater import HMapCLimUpdater
from occupancy_and_vital_signs_detection.core.plot_configurator.plot_updater import PlotUpdater
from occupancy_and_vital_signs_detection.h5_to_image import rolling_average_factory
from occupancy_and_vital_signs_detection.main import Config

logger = logging.getLogger("root")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def main():
    fp = h5py.File("datasets/ds3/h5_out/angle/+30v.packet.20220706T222723.h5")
    config = Config(MMWaveConfigLoader(Path("configs/vod_vs_68xx_10fps_center.cfg").read_text().split("\n")))
    source_it = HMapOnlyMMWInfoIterator(np.asarray(fp['heatmap/full']))

    fig: plt.Figure = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(1, 2)
    plot = (PlotGroupBuilder()
            .with_config(config)
            .set_show_rect_in_hmap(False)
            .add_plot_type(PlotType.POLAR_HMAP, fig.add_subfigure(gs[0]))
            .add_plot_type(PlotType.FULL_HMAP, fig.add_subfigure(gs[1]))
            .build())

    pipeline = PlotConfiguratorPipeline(
        HMapCLimUpdater(rolling_average_factory),
        PlotUpdater()
    )

    try:
        PlotShower(fig, source_it, plot, pipeline, delay=0.107, mute=True).display()
    except KeyboardInterrupt:
        logger.info("interrupted")

    logger.info("done.")


if __name__ == '__main__':
    main()
